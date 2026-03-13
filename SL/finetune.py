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
from validation import validate
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

import os

from peft import LoraConfig, EvaConfig, get_peft_model, initialize_lora_eva_weights

# ========== Config ==========

config_path = "SL/configs/xlstm_large_optimal_config_2.yaml"
state_path = "SL/checkpoints/xlstm_large_optimal_config_2/d2026-01-12|20:36:42_s1000/state.yaml" # state_path=None will initialize a model
device = "cuda"
torch_compile_type = "default"

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
          "model_path": f"./SL/adapters/{c["model_name"]}/d{current_time}_s0/model.pt",
          "optimizer_path": f"./SL/adapters/{c["model_name"]}/d{current_time}_s0/optimizer.pt"}
    state_path = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"

# env_variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_default_device(device) # do this at every needed tensor instead
random.seed(c["seed"])
torch.manual_seed(c["seed"])
torch.backends.cuda.matmul.allow_tf32 = True
# only for convolutional layers (not needed for xlstm)
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

# load tokenizer
tokenizer = Tokenizer.from_file(c["tokenizer_path"])
c["vocab_size"] = tokenizer.get_vocab_size()

# load dataset
dataset = load_from_disk(c["dataset_path"], keep_in_memory=False)

# dataset split (0-indexing)
val_start_idx = math.floor(len(dataset) * (1 - c["validation_size"]))

# start with the next sample
# end index itself is not included
train_dataset = dataset.select(range(sc["trained_sample_idx"] + 1, val_start_idx))
# makes it output pytorch tensors
# select the column of the dataset to format
# device cpu so we can move a full batch to gpu, might also save vram
train_dataset.set_format(type="torch", columns=["text"], device="cpu")
# makes an iterable of dataset that yields dictionary with pytorch tensor [batch_size, block_size] at key "text"
# last batch will be shorter if not divisible by batch size, should not be a problem
train_loader = DataLoader(train_dataset, batch_size=c["hardware_batch_size"], pin_memory=True, num_workers=8)

val_dataset = dataset.select(range(val_start_idx, len(dataset)))
val_dataset.set_format(type="torch", columns=["text"], device="cpu")
val_loader = DataLoader(val_dataset, batch_size=c["hardware_batch_size"], pin_memory=True, num_workers=8)

eva_dataset = dataset.select(range(sc["trained_sample_idx"] + 1, sc["trained_sample_idx"] + 1 + 10000))
eva_dataset.set_format(type="torch", columns=["text"], device="cpu")
# remove last element from each sequence
eva_dataset = eva_dataset.map(lambda x: {"text": x["text"][:-1]})
eva_dataset = eva_dataset.rename_column("text", "x")
eva_loader = DataLoader(eva_dataset, batch_size=c["hardware_batch_size"], pin_memory=True, num_workers=8)

# model
match(c["model_type"]):
    case "transformer":
        bot = modules.transformer(c)
        bot = torch.compile(bot, mode=torch_compile_type)
        # try loading parameters
        try:
            bot.load_state_dict(torch.load(sc["model_path"], weights_only=True))
        except FileNotFoundError:
            print(f"Model {sc["model_path"]} does not exist yet, a new one will be created")

        eva_config = EvaConfig()
        lora_config = LoraConfig( 
            r=16,
            # no embedding, pos_encoding or layernorms
            target_modules=["qkv_projection", "final_projection", "l1", "l2"],
            init_lora_weights="eva",
            eva_config=eva_config
        )
        peft_bot = get_peft_model(bot, lora_config)
        initialize_lora_eva_weights(peft_bot, eva_loader, prepare_model_inputs_fn = None, prepare_layer_inputs_fn = None)
        bot.train()
    case "xlstm":
        # configure the model with TFLA Triton kernels
        xlstm_config = xLSTMLargeConfig(
            embedding_dim=c["d_model"],
            num_heads=c["num_heads"],
            num_blocks=c["num_layers"],
            vocab_size=c["vocab_size"],
            return_last_states=c["return_last_states"],
            chunkwise_kernel=c["chunkwise_kernel"], 
            sequence_kernel=c["sequence_kernel"],
            step_kernel=c["step_kernel"],
            mode="train",
        )
        # instantiate the model
        bot = xLSTMLarge(xlstm_config)
        bot = torch.compile(bot, mode=torch_compile_type)
        # try loading parameters
        try:
            bot.load_state_dict(torch.load(sc["model_path"], weights_only=True))
        except FileNotFoundError:
            print(f"Model {sc["model_path"]} does not exist yet, a new one will be created")

        target_modules = []
        for name, module in bot.named_modules():
            if(name != "embedding" 
               and name.find("norm") == -1 
               and name.find("backend") == -1 
               and name != "lm_head" 
               and name.count(".") == 4
               and name.find("")):
                target_modules.append(name)

        eva_config = EvaConfig()
        lora_config = LoraConfig( 
            r=16,
            # no embedding, pos_encoding or layernorms
            target_modules=["q", "k", "v", "preact", "out_proj", "proj_up_gate", "proj_up", "proj_down"],
            # init_lora_weights="eva",
            # eva_config=eva_config
        )
        peft_bot = get_peft_model(bot, lora_config)
        # initialize_lora_eva_weights(peft_bot, eva_loader, prepare_model_inputs_fn = None, prepare_layer_inputs_fn = None)
        bot.train()
    case _:
        raise ValueError("unknown optimizer")

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
# try:
#     optimizer.load_state_dict(torch.load(sc["optimizer_path"], weights_only=True))
# except FileNotFoundError:
#     print("Optimizer does not exist, a new one will be created")

# if initialized from something, set paths to new directory after loading weights
if(sc["initialized_from"] is not None):
    state_path = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"
    sc["model_path"] = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/model.pt"
    sc["optimizer_path"] = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/optimizer.pt"

# save state config
os.makedirs(f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}")
with open(state_path, "w") as StateConfigFile:
    yaml.safe_dump(sc, StateConfigFile, sort_keys=False)

# save initialization / copy of model that was used for initialization
torch.save(bot.state_dict(), sc["model_path"])
torch.save(optimizer.state_dict(), sc["optimizer_path"])

writer = SummaryWriter(log_dir=f"./SL/runs/{c["model_name"]}_adapters/")
start = time.perf_counter()

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

@torch.compile(mode=torch_compile_type)
def compiled_step(loss):
    optimizer.step()

# ========== Loop ==========

print("-----------------------------")
# start at the next step
step = sc["trained_steps"] + 1
for iteration, batch in enumerate(train_loader, start = sc["trained_steps"] * num_accumulation_steps + 1):

    torch.compiler.cudagraph_mark_step_begin()

    # get batch size and sequence length
    B, S = batch["text"][:,:-1].size()
    # move batch to gpu
    batch["text"] = batch["text"].to(device, non_blocking=True)

    for param in optimizer.param_groups:
        param["lr"] = cosine_anneal(c, step)

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        if(c["model_type"] == "transformer"):
            logits = bot.forward(batch["text"][:,:-1])
        else:
            logits = bot.forward(batch["text"][:,:-1])[0]

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

        if (step % c["steps/eval"] == 0 or step == c["max_steps"]):
            current = time.perf_counter()

            eval_loss, eval_px = validate(bot, val_loader, loss_function, c, device)
            writer.add_scalar("Loss/eval", eval_loss, step)
            writer.add_scalar("Perplexity/eval", eval_px, step)
            writer.flush()

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
            state_path = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/state.yaml"
            sc["model_path"] = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/model.pt"
            sc["optimizer_path"] = f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}/optimizer.pt"

            # make directory to save to
            os.makedirs(f"./SL/adapters/{c["model_name"]}/d{sc["current_time"]}_s{sc["trained_steps"]}")

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
