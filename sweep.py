
import numpy as np
import torch
import os

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb

from train import objective

run_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        # 'open_choice': {
        #     "values": [True], # [True, False],
        # },
        'condition': {
            "values": [
                "Transformer", 
                "Transformer-2xFF", 
                "Transformer-2n_embd", 
                "Transformer-2n_head-2n_embd",
                "Double", 
                "w-Double", 
                "Ours", 
                "Ours-2xT"
            ],
        },
        'n_layer': {
            "values": [6], # [2, 4, 6, 8, 10],
        },
    },
}

def main():

    run = wandb.init(
        project="shakespeare-char",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    # loss, done_when = objective(wandb.config)
    loss = objective(wandb.config, save_checkpoint=False)
    wandb.log({"loss": loss})
    # wandb.log({"done_when": done_when})


if __name__ == "__main__":
    sweep_id = [None, "uhwxos70"][-1]
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=run_configuration, project="shakespeare-char")
    count = 1e4
    wandb.agent(sweep_id, function=main, count=count, project="shakespeare-char")
