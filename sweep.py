
import numpy as np
import torch
import os

import wandb

from train import objective

# # tmp
# run_configuration = {
#     "method": "random",
#     "name": "tmp",
#     "metric": {"goal": "minimize", "name": "loss"},
#     "parameters": {
#         # 'open_choice': {
#         #     "values": [True], # [True, False],
#         # },
#         'condition': {
#             "values": [
#                 # "Transformer", 
#                 # "Transformer-2xFF", 
#                 # "Transformer-2n_embd", 
#                 # "Transformer-2n_head-2n_embd",
#                 # "Double", 
#                 # "w-Double", 
#                 "Ours", 
#                 # "Ours-2xT"
#                 # "Ours-no_ID"
#             ],
#         },
#         'max_router_iter': {
#             "min": 1,
#             "max": 50,
#             "distribution": "q_log_uniform_values",
#             "q": 1,
#         },
#         'n_LSTM_layer': {
#             "values": [2],
#         },
#         'LSTM_hidden_size': {
#             "values": [32],
#         },
#         'n_layer': {
#             "values": [3],
#         },
#         'n_head': {
#             "values": [4],
#         },
#         'n_embd': {
#             "values": [32], 
#         },
#         'dropout': {
#             "values": [0.22], 
#         },
#         'learning_rate': {
#             "values": [0.0005539], 
#         },
#         'max_iters': {
#             "values": [61700],
#         },
#         'intermediate_outputs_training': {
#             "values": [True],
#         },
#     },
# }

# Compare architectures
run_configuration = {
    "method": "random",
    "name": "ours_search",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        # 'open_choice': {
        #     "values": [True], # [True, False],
        # },
        'condition': {
            "values": [
                # "Transformer", 
                # "Transformer-2xFF", 
                # "Transformer-2n_embd", 
                # "Transformer-2n_head-2n_embd",
                # "Double", 
                # "w-Double", 
                "Ours", 
                # "Ours-2xT"
                "Ours-no_ID"
            ],
        },
        'max_router_iter': {
            "min": 1,
            "max": 50,
            "distribution": "q_log_uniform_values",
            "q": 1,
        },
        'n_LSTM_layer': {
            "min": 2,
            "max": 6,
            "distribution": "q_log_uniform_values",
            "q": 1,
        },
        'LSTM_hidden_size': {
            "min": 64,
            "max": 3072,
            "distribution": "q_log_uniform_values",
            "q": 32,
        },
        'n_layer': {
            "values": [6], # [2, 4, 6, 8, 10],
        },
        'n_head': {
            "values": [16], 
        },
        'n_embd': {
            "values": [960], 
        },
        'dropout': {
            "values": [0.22], 
        },
        'learning_rate': {
            "values": [0.0005539], 
        },
        'max_iters': {
            "values": [61700],
        },
        'intermediate_outputs_training': {
            "values": [True],
        },
    },
}

# # Hyperparameter search for transformer
# run_configuration = {
#     "name": "transformer_search",
#     "method": "random", # "bayes",
#     "metric": {"goal": "minimize", "name": "loss"},
#     "parameters": {
#         # 'open_choice': {
#         #     "values": [True], # [True, False],
#         # },
#         'condition': {
#             "values": [
#                 "Transformer", 
#             ],
#         },
#         'n_layer': {
#             "min": 2,
#             "max": 20,
#             "distribution": "q_log_uniform_values",
#             "q": 2,
#         },
#         'n_head': {
#             "values": [1, 2, 4, 8, 16], 
#         },
#         'n_embd': {
#             "min": 64,
#             "max": 3072,
#             "distribution": "q_log_uniform_values",
#             "q": 32,
#         },
#         'dropout': {
#             "min": 0.2,
#             "max": 0.5,
#             "distribution": "log_uniform_values",
#         },
#         'learning_rate': {
#             "min": 1e-5,
#             "max": 1e-3,
#             "distribution": "log_uniform_values",
#         },
#         'max_iters': {
#             "min": 1e3,
#             "max": 1e5,
#             "distribution": "q_log_uniform_values",
#             "q": 50,
#         },
#     },
# }

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
    sweep_id = [None, "yilwx8g5", "52voc92u", "het0u5tt", "e8mjkq03"][0]
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=run_configuration, project="GW_LLM")
    count = 1000
    wandb.agent(sweep_id, function=main, count=count, project="GW_LLM")
