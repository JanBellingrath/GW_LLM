import torch
import os
import yaml

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb
from train import objective


def main():

    # Instead of defining configs here, load them from a yaml file
    with open("configs.yaml", "r") as file:
        configs = yaml.safe_load(file)

    # Launch one run per config
    for config in configs:
        run = wandb.init(project="GW_LLM", config=config)
        run_config = wandb.config
        best_val_loss = objective(run_config, seed=42, out_dir=f"wandb_runs_ckpt/{run.id}-{run.name}")
        run.log({"best_val_loss": best_val_loss})
        run.finish()




if __name__ == "__main__":
    main()
