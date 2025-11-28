import os
os.environ["HF_HOME"] = "/mnt/ssd/huggingface"

from datasets import load_from_disk
from tokenizers import Tokenizer

import yaml

import datetime

import time
import math
import random

import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torcheval.metrics.functional import perplexity

import modules

import os

# ========== Functions ==========

# learning rate decay scheduler (cosine with warmup)
# looks like a slope with the warmup being the ladder
def cosine_anneal(c, step):
    # 1) linear warmup for warmup_iters steps, reaches lr on last warmup step
    if step <= c["warmup_steps"]:
        return c["initial_lr"] * step / (c["warmup_steps"])
    # 2) if it > lr_decay_iters, return min learning rate
    if step >= c["max_steps"]:
        return c["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - c["warmup_steps"]) / (c["max_steps"] - c["warmup_steps"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return c["min_lr"] + coeff * (c["initial_lr"] - c["min_lr"])

def linear_anneal(c, step):
    return c["lr"] - (step - 1) * (c["lr"] / c["max_steps"])

@torch.compile(mode="max-autotune-no-cudagraphs")
def compiled_step(loss):
    optimizer.step()

# ========== Config ==========

config_path = "./SL/configs/fineweb_transformer_max_v8.yaml"
state_path = None # state_path=None will initialize a model

# ========== Init ==========

current_time = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

# open config
with open(config_path) as ConfigFile:
    c = yaml.safe_load(ConfigFile)

# open or generate state config
if(state_path is not None):
    with open(state_path) as StateConfigFile:
        sc = yaml.safe_load(StateConfigFile)
        sc["initialized_from"] = state_path
        sc["current_time"] =  current_time
else:
    sc = {"trained_steps": 0, 
          "current_lr": c["initial_lr"], 
          "trained_sample_idx": -1, 
          "current_time": current_time, 
          "initialized_from": None,
          "model_path": f"./SL/checkpoints/{c["model_name"]}/d{current_time}_s0/model.pt",
          "optimizer_path": f"./SL/checkpoints/{c["model_name"]}/d{current_time}_s0/optimizer.pt"}
    state_path = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"

# env_variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_default_device("cuda") # do this at every tensor instead
random.seed(c["seed"])
torch.manual_seed(c["seed"])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# load tokenizer
tokenizer = Tokenizer.from_file(c["tokenizer_path"])
c["vocab_size"] = tokenizer.get_vocab_size()

# load dataset
dataset = load_from_disk(c["dataset_path"], keep_in_memory=False)
# start with the next sample
dataset = dataset.select(range(sc["trained_sample_idx"] + 1, len(dataset)))
# makes it output pytorch tensors
# select the column of the dataset to format
dataset.set_format(type="torch", columns=["text"], device="cpu")
# makes an iterable of dataset that yields dictionary with pytorch tensor [batch_size, block_size] at key "text"
# last batch will be shorter if not divisible by batch size, should not be a problem
train_loader = DataLoader(dataset, batch_size=c["hardware_batch_size"], pin_memory=True, num_workers=8)  

# model
bot = modules.transformer(c)
bot = torch.compile(bot, mode="max-autotune-no-cudagraphs")
bot.train()
try:
    bot.load_state_dict(torch.load(sc["model_path"], weights_only=True))
except FileNotFoundError:
    print(f"Model {sc["model_path"]} does not exist yet, a new one will be created")

loss_function = torch.nn.CrossEntropyLoss(
    reduction="mean", label_smoothing=c["label_smoothing"])

num_accumulation_steps = c["effective_batch_size"] // c["hardware_batch_size"]

# optimizer
match(c["optimizer"]):
    case "SGD":
        optimizer = torch.optim.SGD(bot.parameters(), lr=c["initial_lr"])
    case "Adam":
        optimizer = torch.optim.AdamW(bot.parameters(), lr=c["initial_lr"], betas=((0.9,0.999))
                                    ,eps=10e-9, fused=True, weight_decay=c["weight_decay"])
    case _:
        raise ValueError("unknown optimizer")
try:
    optimizer.load_state_dict(torch.load(sc["optimizer_path"], weights_only=True))
except FileNotFoundError:
    print("Optimizer does not exist, a new one will be created")

# if initialized from something, set paths to new directory after loading weights
if(sc["initialized_from"] is not None):
    state_path = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"
    sc["model_path"] = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/model.pt"
    sc["optimizer_path"] = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/optimizer.pt"

# save state config
os.makedirs(f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}")
with open(state_path, "w") as StateConfigFile:
    yaml.safe_dump(sc, StateConfigFile, sort_keys=False)

# save initialization / copy of model that was used for initialization
torch.save(bot.state_dict(), sc["model_path"])
torch.save(optimizer.state_dict(), sc["optimizer_path"])

writer = SummaryWriter(log_dir=f"./SL/runs/{c["model_name"]}/")
start = time.perf_counter()

# ========== Loop ==========

print("-----------------------------")
# start at the next step
step = sc["trained_steps"] + 1
for iteration, batch in enumerate(train_loader, start = sc["trained_steps"] * num_accumulation_steps + 1):

    torch.compiler.cudagraph_mark_step_begin()

    # move batch to gpu
    batch["text"] = batch["text"].to("cuda", non_blocking=True)

    for param in optimizer.param_groups:
        param["lr"] = cosine_anneal(c, step)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = bot.forward(batch["text"][:,:-1])

        # get batch size and sequence length
        B, S = batch["text"][:,:-1].size()
        input = logits.reshape(B*S, c["vocab_size"])
        target = batch["text"][:,1:].reshape(B*S)

        # scaled by num_accumulation_steps, so that it stays being the mean for the optimizer
        loss = loss_function(input, target) / num_accumulation_steps

        # compute gradient
        loss.backward()

    # only do a step every num_accumulation_steps iterations
    if(iteration % num_accumulation_steps==0):

        # clip gradients
        torch.nn.utils.clip_grad_norm_(bot.parameters(), 1.0)

        # optimizer step
        compiled_step(loss)

        # flushes gradients from memory as they are not needed anymore
        for param in bot.parameters():
            param.grad = None

        if (step % c["steps/metrics"] == 0 or step == c["max_steps"]):
            current = time.perf_counter()

            # perplexity before current optimizer.step
            px = perplexity(torch.log_softmax(logits, dim=-1), batch["text"][:,1:])

            writer.add_scalar("Loss/step", loss*num_accumulation_steps, step)
            writer.add_scalar("Perplexity/step", px, step)
            writer.add_scalar("time*s", current-start, step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
            writer.flush()

            print(f"Step: {step}")
            print(f"Loss: {loss*num_accumulation_steps}")
            print(f"Perplexity: {px}")
            print(f"Time elapsed: {current-start}s")
            print("-----------------------------")

        if (step % c["steps/save"] == 0 or step == c["max_steps"]):

            sc["trained_steps"] = step 
            sc["current_lr"] = optimizer.param_groups[0]["lr"]
            sc["trained_sample_idx"] = step * c["effective_batch_size"] - 1 # 0-indexing
            # rescaled to mean of the hardware batch
            sc["current_loss"] = loss.item() * num_accumulation_steps
            # current time 
            sc["current_time"] = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

            old_model_path = sc["model_path"]
            old_optimizer_path = sc["optimizer_path"]

            sc["initialized_from"] = state_path
            state_path = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"
            sc["model_path"] = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/model.pt"
            sc["optimizer_path"] = f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/optimizer.pt"

            # make directory to save to
            os.makedirs(f"./SL/checkpoints/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}")

            torch.save(bot.state_dict(), sc["model_path"])
            torch.save(optimizer.state_dict(), sc["optimizer_path"])

            if(c["delete_old_checkpoints"]):
                os.remove(old_model_path)
                os.remove(old_optimizer_path)
        
            with open(state_path, "w") as StateConfigFile:
                yaml.safe_dump(sc, StateConfigFile, sort_keys=False)

        step += 1

    # stops if steps reached or no more batches
    if(step > c["max_steps"]):
        writer.close()
        break
