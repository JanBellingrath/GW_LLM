import torch
import os

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb
from train import objective


def main():

    conditions = ["Transformer", "Big-Transformer", "Double", "w-Double", "Ours"]

    # Launch one run per condition
    for condition in conditions:
        config = {"condition": condition, "n_layer": 6}
        run = wandb.init(project="GW_LLM", config=config)
        run_config = wandb.config
        best_val_loss = objective(run_config, seed=42, out_dir="out-shakespeare-char-"+condition)
        run.log({"best_val_loss": best_val_loss})
        run.finish()




if __name__ == "__main__":
    main()
