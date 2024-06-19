import torch
import os

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb
from sample import sample
from model import GPT, GW_GPT

def main():

    conditions = ["Transformer", "Big-Transformer", "Double", "w-Double", "Ours"]

    for condition in conditions:
        print("Condition: " + condition)
        model_cls = GPT if condition in ["Transformer", "Big-Transformer"] else GW_GPT
        sample(out_dir="out-shakespeare-char-"+condition, model_cls=model_cls)




if __name__ == "__main__":
    main()
