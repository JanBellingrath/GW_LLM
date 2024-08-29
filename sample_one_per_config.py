import torch
import os
import yaml
import sys

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb
from sample import sample
from model import GPT, GW_GPT

def main():

    # Instead of defining configs here, load them from a yaml file
    with open("configs.yaml", "r") as file:
        configs = yaml.safe_load(file)

    out_dir = "babylm/"

    # Sample one run per config
    for config in configs:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        sample(out_dir=out_dir+config["name"], model_cls=model_cls)




if __name__ == "__main__":
    main()
