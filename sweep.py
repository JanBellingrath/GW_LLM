import numpy as np
import torch
import os
import time

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
    try:
        run = wandb.init(
            project="GW_LLM",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        )
        
        try:
            loss = objective(wandb.config, save_checkpoint=False)
            wandb.log({"loss": loss})
        except Exception as e:
            print(f"Error during run execution: {str(e)}")
            # Log the error to wandb
            wandb.log({"error": str(e), "status": "failed"})
            # Return a very high loss value to indicate failure
            return float('inf')
        finally:
            # Ensure we clean up even if there's an error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error during wandb initialization: {str(e)}")
        return float('inf')

def run_sweep():
    """Wrapper function to handle sweep execution with error handling"""
    def wrapped_main():
        try:
            return main()
        except Exception as e:
            print(f"Unhandled exception in sweep run: {str(e)}")
            return float('inf')
    
    return wrapped_main

if __name__ == "__main__":
    sweep_id = [None, "yilwx8g5", "52voc92u", "het0u5tt", "e8mjkq03", "qum883k6"][-1]
    if sweep_id is None:
        try:
            sweep_id = wandb.sweep(config=run_configuration, project="GW_LLM", entity="cerco_neuro_ai")
        except Exception as e:
            print(f"Failed to create sweep: {str(e)}")
            exit(1)
    
    count = 1000
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            wandb.agent(sweep_id, function=run_sweep(), count=count, project="GW_LLM")
            break  # If successful, break out of retry loop
        except Exception as e:
            retry_count += 1
            print(f"Sweep agent failed (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print("Retrying in 60 seconds...")
                time.sleep(60)
            else:
                print("Max retries reached. Exiting.")
                break
