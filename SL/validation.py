import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import perplexity

def validate(bot, validation_loader, loss_function, c, device="cpu"):
    
    losses = []
    perplexities = []

    with torch.no_grad():
        for batch in validation_loader:
            # get batch size and sequence length
            B, S = batch["text"][:,:-1].size()
            # move batch to gpu
            batch["text"] = batch["text"].to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=torch.bfloat16):
                if(c["model_type"] == "transformer"):
                    logits = bot.forward(batch["text"][:,:-1])
                else:
                    logits = bot.forward(batch["text"][:,:-1])[0]

                input = logits.reshape(B*S, c["vocab_size"])
                target = batch["text"][:,1:].reshape(B*S)

                loss = loss_function(input, target)
                px = perplexity(torch.log_softmax(logits, dim=-1), batch["text"][:,1:])
                losses.append(loss)
                perplexities.append(px)

    return torch.tensor(losses).mean(), torch.tensor(perplexities).mean()