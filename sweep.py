
import numpy as np
import torch
import os

import wandb

from train import objective

# # Compare architectures
# run_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "loss"},
#     "parameters": {
#         # 'open_choice': {
#         #     "values": [True], # [True, False],
#         # },
#         'condition': {
#             "values": [
#                 "Transformer", 
#                 "Transformer-2xFF", 
#                 "Transformer-2n_embd", 
#                 "Transformer-2n_head-2n_embd",
#                 "Double", 
#                 "w-Double", 
#                 "Ours", 
#                 "Ours-2xT"
#             ],
#             # "values": [
#             #     "Ours-ACT",
#             # ],
#         },
#         'n_layer': {
#             "values": [6], # [2, 4, 6, 8, 10],
#         },
#     },
# }

# Hyperparameter search for transformer
run_configuration = {
    "name": "transformer_search",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        # 'open_choice': {
        #     "values": [True], # [True, False],
        # },
        'condition': {
            "values": [
                "Transformer", 
            ],
        },
        'n_layer': {
            "min": 2,
            "max": 20,
            "distribution": "q_log_uniform_values",
            "q": 1,
        },
        'n_head': {
            "values": [1, 2, 4, 8, 16], 
        },
        'n_embd': {
            "min": 64,
            "max": 768,
            "distribution": "q_log_uniform_values",
            "q": 32,
        },
        'dropout': {
            "min": 0.2,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        'learning_rate': {
            "min": 1e-5,
            "max": 1e-3,
            "distribution": "log_uniform_values",
        },
    },
}

def main():

    run = wandb.init(
        project="GW_LLM",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    # loss, done_when = objective(wandb.config)
    loss = objective(wandb.config, save_checkpoint=False)
    wandb.log({"loss": loss})
    # wandb.log({"done_when": done_when})


if __name__ == "__main__":
    sweep_id = [None, "yilwx8g5", "52voc92u", "uauevnup"][-1]
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=run_configuration, project="GW_LLM")
    count = 1000
    wandb.agent(sweep_id, function=main, count=count, project="GW_LLM")
