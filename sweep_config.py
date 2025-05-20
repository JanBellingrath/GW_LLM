import wandb

def create_sweep():
    """
    Create a W&B sweep for tuning router and training hyperparameters using Bayesian optimization
    with ASHA early-stopping. Assumes the training script handles `lr_schedule` to set up
    appropriate schedulers (e.g., linear warmup + cosine decay).
    """
    print("\n🔧 Configuring sweep parameters...")
    sweep_config = {
        'program': 'train.py',  # specify the training script for W&B agents to run
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'num_layers':       {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
            'hidden_dim':       {'values': [16, 32, 64]},
            'dropout':          {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            'learning_rate':    {
                                    'distribution': 'log_uniform',
                                    'min': 1e-6,
                                    'max': 1e-2
                                },
            'batch_size':       {'values': [16, 32, 64, 128, 256]},
            'grad_clip_norm':   {'values': [0.1, 0.5, 1.0, 5.0]},
            'weight_decay':     {'values': [0.0, 1e-6, 1e-5, 1e-4]},
            'softmax_temperature': {'values': [0.5, 1.0, 2.0]},
            'lr_schedule':      {'values': ['constant', 'linear_warmup_cosine', 'cosine']},
            'normalize_router_logits': {'values': [False, True]},
            'normalize_residual_stream': {'values': [False, True]},
            # Temperature annealing strategy
            'temperature_schedule': {
                'values': ['constant', 'linear', 'exponential']
            },
            # Initial and final temps for annealing
            'temperature_initial': {
                'distribution': 'log_uniform',
                'min': 0.1,
                'max': 5.0
            },
            'temperature_final': {
                'distribution': 'log_uniform',
                'min': 0.01,
                'max': 1.0
            },
            # Controls speed of annealing (fraction of total iters over which to anneal)
            'temperature_anneal_fraction': {
                'values': [0.25, 0.5, 0.75]
            },
        },
        'early_terminate': {
            'type': 'asha',
            'min_iter': 5,
            'eta': 2,
            's': 0
        }
    }
    
    print("📊 Registering sweep with Weights & Biases...")
    sweep_id = wandb.sweep(sweep_config, project='GW_LLM', entity='cerco_neuro_ai')
    print(f"✨ Created new sweep with ID: {sweep_id}")
    return sweep_id

if __name__ == '__main__':
    print("\n🚀 === Starting Sweep Configuration === 🚀")
    try:
        sweep_id = create_sweep()
        print(f"✅ Sweep configuration complete! ID: {sweep_id}")
    except Exception as e:
        print(f"\n❌ Error creating sweep: {type(e).__name__}: {str(e)}")
        raise

"""
W&B sweep configuration for tuning router and training hyperparameters using Bayesian optimization
with ASHA early-stopping. Assumes the training script handles `lr_schedule` to set up
appropriate schedulers (e.g., linear warmup + cosine decay).
"""

sweep_configuration = {
    'program': 'train.py',  # specify the training script for W&B agents to run
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'num_layers':       {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
        'hidden_dim':       {'values': [16, 32, 64]},
        'dropout':          {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'learning_rate':    {
                                'distribution': 'log_uniform',
                                'min': 1e-6,
                                'max': 1e-2
                            },
        'batch_size':       {'values': [16, 32, 64, 128, 256]},
        'grad_clip_norm':   {'values': [0.1, 0.5, 1.0, 5.0]},
        'weight_decay':     {'values': [0.0, 1e-6, 1e-5, 1e-4]},
        'softmax_temperature': {'values': [0.5, 1.0, 2.0]},
        'lr_schedule':      {'values': ['constant', 'linear_warmup_cosine', 'cosine']},
        'normalize_router_logits': {'values': [False, True]},
        'normalize_residual_stream': {'values': [False, True]},
        # Temperature annealing strategy
        'temperature_schedule': {
            'values': ['constant', 'linear', 'exponential']
        },
        # Initial and final temps for annealing
        'temperature_initial': {
            'distribution': 'log_uniform',
            'min': 0.1,
            'max': 5.0
        },
        'temperature_final': {
            'distribution': 'log_uniform',
            'min': 0.01,
            'max': 1.0
        },
        # Controls speed of annealing (fraction of total iters over which to anneal)
        'temperature_anneal_fraction': {
            'values': [0.25, 0.5, 0.75]
        },
    },
    'early_terminate': {
        'type': 'asha',
        'min_iter': 5,
        'eta': 2,
        's': 0
    }
} 