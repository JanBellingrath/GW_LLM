#!/usr/bin/env python
# Add these at the very top of train.py, before anything else
import sys
sys.modules['__main__'].__builtins__.__dict__['__debug__'] = False

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

print("🚀 === Starting Training Script === 🚀")
import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import argparse
print("✨ Basic imports done")

from PIL import Image
import numpy as np
import torch
print("📦 NumPy and Torch imported")

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
print("⚡ Torch DDP imported")

from model import GPTConfig, GPT, GW_GPT
from router_lstm import RouterLSTM
from routing_analysis import routing_analysis
from sample import sample
print("🔧 Local modules imported")

import wandb
print("📊 WandB imported")
# wandb.login(key="840819a79d14bdbf926b4317947f6fc499966e8c")

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"  # Keeping this commented line for debugging purposes
print(f"\n💻 Using device: {DEVICE}")

# Set PYTORCH_CUDA_ALLOC_CONF to prevent memory fragmentation as mentioned in the error
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable memory history tracking if available
def enable_memory_tracking():
    """Enable PyTorch CUDA memory history tracking if available."""
    if torch.cuda.is_available() and hasattr(torch.cuda.memory, '_record_memory_history'):
        print("🔍 Enabling CUDA memory history tracking...")
        torch.cuda.memory._record_memory_history(max_entries=10000)
        return True
    else:
        print("⚠️ CUDA memory history tracking not available")
        return False

# Function to dump memory snapshot and upload to wandb
def dump_memory_snapshot_to_wandb(snapshot_name, step=None):
    """Dump a memory snapshot and upload to wandb."""
    if not torch.cuda.is_available() or not hasattr(torch.cuda.memory, '_dump_snapshot'):
        print("⚠️ Memory snapshot dumping not available")
        return
    
    print(f"📸 Taking memory snapshot: {snapshot_name}")
    
    # Create snapshots directory if it doesn't exist
    os.makedirs('snapshots', exist_ok=True)
    
    # Clean up old snapshots if there are too many (keep only last 5)
    snapshot_files = sorted(os.listdir('snapshots'))
    if len(snapshot_files) > 5:
        for old_file in snapshot_files[:-5]:
            try:
                os.remove(os.path.join('snapshots', old_file))
                print(f"🗑️ Removed old snapshot: {old_file}")
            except Exception as e:
                print(f"⚠️ Error removing old snapshot: {e}")
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_path = f"snapshots/{snapshot_name}_{timestamp}.pickle"
    
    # Dump the snapshot
    print(f"💾 Saving memory snapshot to {snapshot_path}...")
    torch.cuda.memory._dump_snapshot(snapshot_path)
    
    # Upload to wandb
    if wandb.run is not None:
        print("📤 Uploading snapshot to WandB...")
        wandb.save(snapshot_path, base_path="snapshots/")
        
        # Log snapshot as an artifact
        artifact = wandb.Artifact(f"memory_snapshot_{snapshot_name}", type="memory_snapshot")
        artifact.add_file(snapshot_path)
        wandb.log_artifact(artifact, aliases=[f"step_{step}" if step is not None else "latest"])
        
        # Analyze and log memory snapshot details
        log_memory_analysis_to_wandb(snapshot_path, step=step)
        print("✅ Memory snapshot processed and uploaded")
        
    return snapshot_path

def analyze_memory_snapshot(snapshot_path):
    """
    Analyze a memory snapshot and return a summary of the largest tensors.
    
    This function opens a snapshot file created by torch.cuda.memory._dump_snapshot,
    analyzes it to find the largest tensors, and returns a summary.
    """
    if not os.path.exists(snapshot_path):
        print(f"Error: Snapshot file {snapshot_path} does not exist.")
        return None
    
    try:
        # Load the snapshot
        with open(snapshot_path, 'rb') as f:
            snapshot = pickle.load(f)
        
        # Extract tensor sizes and sort by size
        tensors = []
        
        # The exact structure depends on PyTorch version, try to handle different formats
        if isinstance(snapshot, dict) and 'tensors' in snapshot:
            # Newer PyTorch versions
            for tensor_info in snapshot['tensors']:
                size_bytes = tensor_info.get('size', 0)
                shape = tensor_info.get('shape', [])
                dtype = tensor_info.get('dtype', 'unknown')
                allocation_site = tensor_info.get('allocation_site', 'unknown')
                
                tensors.append({
                    'size_mb': size_bytes / (1024 * 1024),
                    'shape': shape,
                    'dtype': dtype,
                    'allocation_site': allocation_site
                })
        elif hasattr(snapshot, 'items'):
            # Fallback for other formats
            for key, value in snapshot.items():
                if isinstance(value, dict) and 'size' in value:
                    size_bytes = value.get('size', 0)
                    shape = value.get('shape', [])
                    dtype = value.get('dtype', 'unknown')
                    allocation_site = value.get('allocation_site', 'unknown')
                    
                    tensors.append({
                        'size_mb': size_bytes / (1024 * 1024),
                        'shape': shape,
                        'dtype': dtype,
                        'allocation_site': allocation_site
                    })
        
        # Sort tensors by size (largest first)
        tensors.sort(key=lambda x: x['size_mb'], reverse=True)
        
        # Get top 50 largest tensors
        largest_tensors = tensors[:50]
        
        # Create a summary
        total_memory_mb = sum(tensor['size_mb'] for tensor in tensors)
        top_tensors_memory_mb = sum(tensor['size_mb'] for tensor in largest_tensors)
        
        summary = {
            'snapshot_file': snapshot_path,
            'total_memory_mb': total_memory_mb,
            'top_tensors_memory_mb': top_tensors_memory_mb,
            'percent_in_top_tensors': (top_tensors_memory_mb / total_memory_mb * 100) if total_memory_mb > 0 else 0,
            'num_tensors': len(tensors),
            'largest_tensors': largest_tensors
        }
        
        return summary
    
    except Exception as e:
        print(f"Error analyzing memory snapshot: {e}")
        import traceback
        traceback.print_exc()
        return None

def log_memory_analysis_to_wandb(snapshot_path, step=None):
    """Analyze memory snapshot and log results to wandb."""
    if not wandb.run:
        print("WandB run not initialized, skipping memory analysis logging")
        return
    
    summary = analyze_memory_snapshot(snapshot_path)
    if not summary:
        return
    
    # Create a table for the largest tensors
    table = wandb.Table(columns=["Size (MB)", "Shape", "Dtype", "Allocation Site"])
    
    for tensor in summary['largest_tensors'][:20]:  # Log top 20 tensors
        table.add_data(
            f"{tensor['size_mb']:.2f}",
            str(tensor['shape']),
            str(tensor['dtype']),
            str(tensor['allocation_site'])
        )
    
    # Log the summary stats and table
    wandb.log({
        f"memory/snapshot_summary": {
            "total_memory_mb": summary['total_memory_mb'],
            "top_tensors_memory_mb": summary['top_tensors_memory_mb'],
            "percent_in_top_tensors": summary['percent_in_top_tensors'],
            "num_tensors": summary['num_tensors'],
        },
        f"memory/largest_tensors": table
    }, step=step)
    
    print(f"Logged memory analysis to wandb for snapshot {snapshot_path}")
    
    return summary

# Enable memory tracking at the start
memory_tracking_enabled = enable_memory_tracking()

def print_memory_stats(current_iter):
    """Print memory statistics with improved visibility but maintaining exact same functionality"""
    # Get memory statistics
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    reserved_memory = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_reserved = torch.cuda.max_memory_reserved() / 1024**2
    
    # Create memory log message with more visible separators
    memory_log = (
        "\n\n" + "!"*80 + "\n"  # Keeping original separator for consistency
        f"📊 ITER {current_iter} - CUDA MEMORY SUMMARY:\n"
        f"ALLOCATED: {allocated_memory:.2f} MB\n"
        f"RESERVED:  {reserved_memory:.2f} MB\n"
        f"MAX ALLOCATED: {max_allocated:.2f} MB\n"
        f"MAX RESERVED:  {max_reserved:.2f} MB\n"
    )
    
    # Calculate growth if possible
    growth = 0.0  # Initialize growth with a default value
    if hasattr(print_memory_stats, "last_allocated"):
        growth = allocated_memory - print_memory_stats.last_allocated
        growth_emoji = "🔺" if growth > 0 else "🔻" if growth < 0 else "➡️"
        memory_log += f"MEMORY GROWTH: {growth:.2f} MB {growth_emoji}\n"
    print_memory_stats.last_allocated = allocated_memory
    
    memory_log += "!"*80 + "\n\n"  # Keeping original separator for consistency
    
    # Write to log file and print to console
    with open('memory_log.txt', 'a') as f:
        f.write(memory_log)
    
    # Force console output even if tqdm is active
    print(memory_log, flush=True)
    sys.stderr.write(memory_log)  # Also write to stderr which might not be buffered
    sys.stderr.flush()
    sys.stdout.flush()
    
    # Return memory stats so they can be used elsewhere
    return {
        "allocated": allocated_memory,
        "reserved": reserved_memory,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
        "growth": growth
    }

import gc
def safe_cuda_cache_clear():
    """
    Safely clear CUDA cache and return amount freed in MB
    
    This function calls torch.cuda.empty_cache() which releases all unused 
    cached memory from PyTorch so it can be used by other GPU applications.
    It doesn't reduce actual allocated memory used by PyTorch objects like tensors,
    but helps with:
    1. Memory fragmentation (by releasing unused blocks)
    2. Making more memory available for other operations
    3. Preventing eventual OOM errors that occur due to fragmentation
    
    Note that tensors that are still referenced will not be freed.
    """
    if torch.cuda.is_available():
        # Run garbage collector first to destroy any cycles
        gc.collect()
        before = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1024**2
        freed = max(0, before - after)  # this can be negative due to timing issues
        if freed > 10:  # Only show message for significant memory clearing
            print(f"🧹 Cleared {freed:.1f}MB of CUDA memory")
        return freed
    return 0

def objective(run_config, seed=None, out_dir=None, save_checkpoint=True, gen_samples=True):
    print("\n🎯 === Starting Training === 🎯")
    print(f"💻 Device: {DEVICE}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📊 Data directory exists: {os.path.exists(os.path.join('data', 'openwebtext'))} 🔍")
    
    # Initialize config with defaults
    config = {}

    # Model core settings - using GPT-2 small as base
    config["init_from"] = 'gpt2'
    config["model_cls"] = GW_GPT
    config["n_layer"] = 12
    config["n_head"] = 12
    config["n_embd"] = 768
    config["expert_k"] = 12  # Allow selecting from all layers
    config["max_router_iter"] = 12  # Route once per layer
    config["open_choice"] = True
    config["weigh_experts"] = True
    config["identity_layers"] = False
    config["dropout"] = 0.0  # No dropout since we're only training router
    config["bias"] = False  # Match GPT-2 settings
    config["block_size"] = 64  # Reduced from 128 to save memory
    config["expansion_factor"] = 4  # Standard MLP expansion

    # Training settings - these will be overridden by sweep config
    config["batch_size"] = 4  # Reduced from 8 to save memory
    config["gradient_accumulation_steps"] = 1
    config["learning_rate"] = 1e-3  # Higher LR since we're only training router
    config["weight_decay"] = 1e-1
    config["beta1"] = 0.9
    config["beta2"] = 0.95
    config["grad_clip"] = 1.0
    
    # Learning rate schedule
    config["decay_lr"] = True
    config["warmup_iters"] = 100
    config["max_iters"] = 2000
    config["lr_decay_iters"] = 2000  # Match max_iters
    config["min_lr"] = 1e-4  # LR/10
    config["lr_schedule"] = "linear_warmup_cosine"  # Default value

    # Router settings - these will be overridden by sweep config
    config["router_num_layers"] = 2  # Default LSTM num_layers
    config["router_hidden_dim"] = 1024  # Default LSTM hidden_dim
    config["router_dropout"] = 0.0  # Default dropout rate
    config["softmax_temperature"] = 1.0  # Default softmax temperature

    # Evaluation and logging
    config["eval_interval"] = 100
    config["eval_iters"] = 50  # Reduced from 200 to save memory
    config["log_interval"] = 10
    config["always_save_checkpoint"] = True
    config["always_analyze_routing"] = True
    config["eval_only"] = False

    # System settings
    config["device"] = DEVICE
    config["dtype"] = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    config["backend"] = 'nccl'
    config["memory_snapshot_interval"] = 200  # Take snapshots every 200 iterations

    # Wandb settings
    config["wandb_log"] = True
    config["wandb_project"] = 'GW_LLM'
    config["wandb_run_name"] = 'gpt2-router'
    
    # Dataset settings
    config["dataset"] = 'openwebtext'  # Changed from babyLM to openwebtext

    # Output directory
    config["out_dir"] = 'out-gpt2-router' if out_dir is None else out_dir

    # Override with WandB sweep config parameters
    # Apply sweep config parameters if they exist
    if wandb.run is not None:
        cfg = wandb.config
        print("WandB sweep config detected, applying parameters:")
        
        # Router params
        if hasattr(cfg, 'num_layers'):
            config["router_num_layers"] = cfg.num_layers
            print(f"  router_num_layers: {cfg.num_layers}")
            
        if hasattr(cfg, 'hidden_dim'):
            config["router_hidden_dim"] = cfg.hidden_dim
            print(f"  router_hidden_dim: {cfg.hidden_dim}")
            
        if hasattr(cfg, 'dropout'):
            config["router_dropout"] = cfg.dropout
            print(f"  router_dropout: {cfg.dropout}")
        
        # Training params
        if hasattr(cfg, 'learning_rate'):
            config["learning_rate"] = cfg.learning_rate
            config["min_lr"] = cfg.learning_rate / 10  # Scale min_lr accordingly
            print(f"  learning_rate: {cfg.learning_rate}")
            
        if hasattr(cfg, 'batch_size'):
            config["batch_size"] = cfg.batch_size
            print(f"  batch_size: {cfg.batch_size}")
            
        if hasattr(cfg, 'weight_decay'):
            config["weight_decay"] = cfg.weight_decay
            print(f"  weight_decay: {cfg.weight_decay}")
            
        if hasattr(cfg, 'grad_clip_norm'):
            config["grad_clip"] = cfg.grad_clip_norm
            print(f"  grad_clip: {cfg.grad_clip_norm}")
            
        if hasattr(cfg, 'softmax_temperature'):
            config["softmax_temperature"] = cfg.softmax_temperature
            print(f"  softmax_temperature: {cfg.softmax_temperature}")
            
        if hasattr(cfg, 'lr_schedule'):
            config["lr_schedule"] = cfg.lr_schedule
            print(f"  lr_schedule: {cfg.lr_schedule}")
    else:
        # Apply specific overrides from run_config for non-sweep runs
        if "learning_rate" in run_config:
            config["learning_rate"] = run_config["learning_rate"]
            config["min_lr"] = config["learning_rate"] / 10

    # Initialize wandb BEFORE training starts
    if config["wandb_log"] and wandb.run is None:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config,
            entity="cerco_neuro_ai"
        )

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config["gradient_accumulation_steps"] % ddp_world_size == 0
        config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = config["gradient_accumulation_steps"] * ddp_world_size * config["batch_size"] * config["block_size"]
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in config["device"] else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config["dtype"]]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    data_dir = os.path.join('data', config["dataset"])
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
        x = torch.stack([torch.from_numpy((data[i:i+config["block_size"]]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config["block_size"]]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(config["device"], non_blocking=True), y.pin_memory().to(config["device"], non_blocking=True)
        else:
            x, y = x.to(config["device"]), y.to(config["device"])
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=config["n_layer"], max_router_iter=config["max_router_iter"], identity_layers=config["identity_layers"], intermediate_outputs_training=False, expert_k=config["expert_k"], open_choice=config["open_choice"], weigh_experts=config["weigh_experts"], n_head=config["n_head"], n_embd=config["n_embd"], block_size=config["block_size"],
                    bias=config["bias"], vocab_size=None, dropout=config["dropout"], expansion_factor=config["expansion_factor"],
                    router_num_layers=config["router_num_layers"], router_hidden_dim=config["router_hidden_dim"], 
                    router_dropout=config["router_dropout"], softmax_temperature=config["softmax_temperature"])
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = config["model_cls"](gptconf)

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args['block_size'] = config["block_size"] # so that the checkpoint will have the right value
    
    # Try to compile the model if torch version supports it (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            print("PyTorch 2.0+ detected - using torch.compile for performance optimization")
            compile_mode = 'max-autotune'  # Options: 'default', 'reduce-overhead', 'max-autotune'
            model = torch.compile(model, mode=compile_mode)
            print(f"Model compiled successfully with mode: {compile_mode}")
        else:
            print("PyTorch version does not support torch.compile - skipping compilation")
    except Exception as e:
        print(f"Model compilation failed with error: {str(e)}")
        print("Continuing with uncompiled model")
    
    model.to(config["device"])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(config["dtype"] == 'float16'))

    # Take a memory snapshot before training starts
    if memory_tracking_enabled and master_process:
        dump_memory_snapshot_to_wandb("before_training", step=0)
    
    optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"], (config["beta1"], config["beta2"]), device_type)
    checkpoint = None # free up memory

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(analyze_routing=False, step=None):
        out = {}
        model.eval()
        
        if memory_tracking_enabled and master_process:
            print(f"📸 Taking memory snapshot at evaluation start (step {step})")
            dump_memory_snapshot_to_wandb(f"start_eval_iter_{step}", step=step)

        freed = safe_cuda_cache_clear()
        print(f"🧹 Cleared {freed:.1f}MB before evaluation")
        
        for split in ['train', 'val']:
            routing_weights = []
            losses = {}
            
            print(f"📊 Evaluating {split} split...")
            
            for k in range(config["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    if analyze_routing:
                        logits, loss, batch_routing_weights = model(X, Y, return_routing_weights=True)
                        routing_weights.extend(batch_routing_weights)
                    else:
                        logits, loss = model(X, Y)

                for key, value in loss.items():
                    if key not in losses:
                        losses[key] = torch.zeros(config["eval_iters"], requires_grad=False)
                    losses[key][k] = value.detach().item()
                
                if k % 10 == 0:
                    print(f"  ⏳ Progress: {k}/{config['eval_iters']} - loss: {loss['total'].item():.4f}")
                    freed = safe_cuda_cache_clear()
                    if freed > 0:
                        print(f"  🧹 Cleared {freed:.1f}MB during evaluation")

            print(f"Finished evaluating {split}")

            out[split] = {key: value.mean().item() for key, value in losses.items()}
            
            # Clear losses dict explicitly
            for k in list(losses.keys()):
                del losses[k]
            del losses
            
            if analyze_routing:
                routing_weights = np.array(routing_weights)

                routing_analysis(routing_weights, dir=os.path.join(wandb.run.dir, split, 'all'))

                routing_weights_shape = routing_weights.shape                    
                routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])
                for i in range(routing_weights.shape[0]):
                    topval = np.sort(routing_weights[i])[-config["expert_k"]]
                    routing_weights[i] = np.where(routing_weights[i] < topval, 0, routing_weights[i])
                routing_weights /= routing_weights.sum(axis=-1, keepdims=True)
                routing_weights = routing_weights.reshape(routing_weights_shape)

                routing_analysis(routing_weights, dir=os.path.join(wandb.run.dir, split, 'topk'))

                routing_table_data = [
                    [
                        "all", 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'all', "layer_mean.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'all', "layer_iter.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'all', "layer_layer.png")))),
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'all', "layer_token.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'all', "layer_seq.png")))),
                    ],
                    [
                        "topk", 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'topk', "layer_mean.png")))),
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'topk', "layer_iter.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'topk', "layer_layer.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'topk', "layer_token.png")))), 
                        wandb.Image(np.asarray(Image.open(
                            os.path.join(wandb.run.dir, split, 'topk', "layer_seq.png")))),
                    ]
                ]
                routing_table = wandb.Table(data=routing_table_data, columns=["data", "layer_mean", "layer_iter", "layer_layer", "layer_token", "layer_seq"])
                wandb.log({split+"/routing": routing_table}, step=step+1)

                images = {}
                for selection in ["all", "topk"]:
                    for data_type in ["layer_layer", "layer_iter", "layer_token", "layer_seq", "layer_mean"]:
                        images["/".join([split, selection, data_type])] = wandb.Image(os.path.join(wandb.run.dir, split, selection, f"{data_type}.png"))
                wandb.log(images, step=step+1)

                #print("After topk - selected_experts range:", selected_experts.min().item(), selected_experts.max().item())
                #print("expert_mask shape:", expert_mask.shape)
                #print("expert_dim_ind range:", expert_dim_ind.min().item(), expert_dim_ind.max().item())
        model.train()
        # Clear cache after evaluation
        safe_cuda_cache_clear()
        
        # Take a memory snapshot at the end of evaluation
        if memory_tracking_enabled and master_process:
            dump_memory_snapshot_to_wandb(f"end_eval_iter_{step}", step=step)
            
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # Check what type of schedule to use
        if config["lr_schedule"] == "constant":
            return config["learning_rate"]
        
        # For linear_warmup_cosine (default) and cosine schedules:
        # 1) linear warmup for warmup_iters steps
        if it < config["warmup_iters"] and config["lr_schedule"] == "linear_warmup_cosine":
            return config["learning_rate"] * it / config["warmup_iters"]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config["lr_decay_iters"]:
            return config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - (config["warmup_iters"] if config["lr_schedule"] == "linear_warmup_cosine" else 0)) / (config["lr_decay_iters"] - (config["warmup_iters"] if config["lr_schedule"] == "linear_warmup_cosine" else 0))
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0

    # Initialize a simple progress tracker without tqdm
    print("\n=== Starting Training Loop ===")
    print(f"Will run for {config['max_iters']} iterations")
    start_time = time.time()

    while True:
        # Simple progress tracking
        if iter_num > 0:  # Skip first iteration
            elapsed = time.time() - start_time
            est_total_time = elapsed * config["max_iters"] / iter_num
            est_remaining = est_total_time - elapsed
            hours_remaining = int(est_remaining // 3600)
            mins_remaining = int((est_remaining % 3600) // 60)
            
            # Print a progress line
            progress_pct = iter_num / config["max_iters"] * 100
            loss_val = loss.item() if 'loss' in locals() else 0.0
            print(f"\r[{progress_pct:.1f}%] Iter {iter_num}/{config['max_iters']} - Loss: {loss_val:.4f} - LR: {lr:.6f} - ETA: {hours_remaining}h {mins_remaining}m", end="", flush=True)
        
        # First get learning rate
        lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # evaluate the loss on train/val sets and write checkpoints
        if (iter_num % config["eval_interval"] == 0 and master_process) or iter_num == config["max_iters"]:
            analyze_routing = (iter_num == config["max_iters"] or config["always_analyze_routing"]) and config["model_cls"] == GW_GPT
            losses = estimate_loss(analyze_routing=analyze_routing, step=iter_num)
            print(f"step {iter_num}: train loss {losses['train']['total']:.4f}, val loss {losses['val']['total']:.4f}")
            if config["wandb_log"]:
                # Get memory metrics for logging
                allocated_memory = torch.cuda.memory_allocated() / 1024**2
                reserved_memory = torch.cuda.memory_reserved() / 1024**2
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                max_reserved = torch.cuda.max_memory_reserved() / 1024**2
                
                wandb_log = {
                    "iter": iter_num,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "memory/allocated_mb": allocated_memory,
                    "memory/reserved_mb": reserved_memory,
                    "memory/max_allocated_mb": max_allocated,
                    "memory/max_reserved_mb": max_reserved,
                }
                for split in ['train', 'val']:
                    for key, value in losses[split].items():
                        wandb_log[f"{split}/{key}"] = value
                wandb.log(wandb_log, step=iter_num+1)

            if losses['val']['total'] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses['val']['total']
                if iter_num > 0 and save_checkpoint:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
                    safe_cuda_cache_clear()  # Clear cache after saving checkpoint
                    
                    # Take a memory snapshot after saving checkpoint
                    if memory_tracking_enabled and master_process:
                        dump_memory_snapshot_to_wandb(f"post_checkpoint_iter_{iter_num}", step=iter_num)

                # We now want to sample a few sentences from the model
                if gen_samples:
                    model.eval()
                
                    # load enc from enc.pkl in data_dir
                    with open(os.path.join(data_dir, 'enc.pkl'), 'rb') as f:
                        enc = pickle.load(f)
                    
                    prompts, generations = sample(wandb.run.dir, model=model, enc=enc, write_to_file=False)

                    gen_table = wandb.Table(columns=["Prompt", "Generation"])
                    for i in range(len(prompts)):
                        gen_table.add_data(prompts[i], generations[i])

                    wandb.log({"generations": gen_table}, step=iter_num+1)

                    model.train()
                    safe_cuda_cache_clear()  # Clear cache after generating samples
                
        if iter_num == 0 and config["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(config["gradient_accumulation_steps"]):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)
            
            # Use autocast for both forward and backward
            with ctx:
                logits, loss = model(X, Y)
                loss = loss['total']
                loss = loss / config["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                
                # Take memory snapshot before backward pass on the last micro step
                is_last_micro_step = (micro_step == config["gradient_accumulation_steps"] - 1)
                if memory_tracking_enabled and master_process and is_last_micro_step and iter_num % config["memory_snapshot_interval"] == 0:
                    dump_memory_snapshot_to_wandb(f"before_backward_iter_{iter_num}", step=iter_num)
                    
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            
            # Take memory snapshot after backward pass on the last micro step
            if memory_tracking_enabled and master_process and is_last_micro_step and iter_num % config["memory_snapshot_interval"] == 0:
                dump_memory_snapshot_to_wandb(f"after_backward_iter_{iter_num}", step=iter_num)
                
        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # Take memory snapshot after optimization step 
        if memory_tracking_enabled and master_process and iter_num % config["memory_snapshot_interval"] == 0:
            dump_memory_snapshot_to_wandb(f"after_optim_iter_{iter_num}", step=iter_num)

        # Aggressively clear cache after backward pass and optimization
        freed_mb = safe_cuda_cache_clear()
        if freed_mb > 10 and iter_num % 10 == 0:  # Only log significant cache clearing
            print(f"Freed {freed_mb:.1f}MB CUDA memory after optimizer step", flush=True)

        # Track memory explicitly
        print("\n\n")  # Ensure we're on a new line after the progress indicator
        print("CHECKING MEMORY USAGE...", flush=True)
        print_memory_stats(iter_num)
        torch.cuda.empty_cache()  # Try to release memory

        # Add memory tracking at regular intervals too
        if iter_num % 10 == 0:  # Track memory every 10 iterations
            print(f"\nMEMORY CHECK at iter {iter_num}: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated\n", flush=True)

        iter_num += 1
        local_iter_num += 1
        
        # termination conditions
        if iter_num > config["max_iters"]:
            # Take a memory snapshot at the end of training
            if memory_tracking_enabled and master_process:
                dump_memory_snapshot_to_wandb("end_of_training", step=iter_num)
            break

    if ddp:
        destroy_process_group()
    
    # End of training loop
    print("\n✅ === Training Complete === ✅")
    return best_val_loss

if __name__ == "__main__":
    print("\n🚀 === Script Starting === 🚀")
    try:
        print(f"💻 Checking CUDA availability: {torch.cuda.is_available()}")
        print(f"📂 Current directory: {os.getcwd()}")
        
        # Parse any command line arguments
        parser = argparse.ArgumentParser(description='Train a GPT model with router')
        parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
        args, _ = parser.parse_known_args()
        
        # Initialize wandb to get sweep config
        if not args.no_wandb:
            print("🔄 Initializing WandB to read sweep configuration...")
            wandb.init()  
            print("✅ WandB initialized successfully")
        
        data_dir = os.path.join('data', 'openwebtext')
        print(f"📂 Checking data files:")
        for file in ['train.bin', 'val.bin', 'enc.pkl']:
            path = os.path.join(data_dir, file)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            status = "✅" if exists else "❌"
            print(f"{status} {file}: size={size/1e9:.2f}GB")
        
        print("\n🎯 === Starting Training === 🎯")
        objective({})
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()