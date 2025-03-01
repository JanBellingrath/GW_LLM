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

import os
import time
import math
import pickle
from contextlib import nullcontext

from PIL import Image

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, GW_GPT
from routing_analysis import routing_analysis
from sample import sample

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

import wandb



def objective(run_config, seed=None, out_dir=None, save_checkpoint=True, gen_samples=True):

    config = {}

    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    config["eval_interval"] = 2000
    config["log_interval"] = 1
    config["eval_iters"] = 200
    config["eval_only"] = False # if True, script exits right after the first eval
    config["always_save_checkpoint"] = False # if True, always save a checkpoint after each eval
    config["init_from"] = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # model_cls = GW_GPT # GPT or GW_GPT
    # data
    config["dataset"] = 'openwebtext'
    config["gradient_accumulation_steps"] = 5 * 8 # used to simulate larger batch sizes
    config["batch_size"] = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    config["block_size"] = 1024
    # model
    config["n_layer"] = 12
    config["max_router_iter"] = 99
    config["expert_k"] = 99
    config["n_head"] = 12
    config["n_embd"] = 768
    config["dropout"] = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    config["bias"] = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    config["learning_rate"] = 6e-4 # max learning rate
    config["max_iters"] = 600000 # total number of training iterations
    config["weight_decay"] = 1e-1
    config["beta1"] = 0.9
    config["beta2"] = 0.95
    config["grad_clip"] = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    config["decay_lr"] = True # whether to decay the learning rate
    config["warmup_iters"] = 2000 # how many steps to warm up for
    config["lr_decay_iters"] = 600000 # should be ~= max_iters per Chinchilla
    config["min_lr"] = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    config["backend"] = 'nccl' # 'nccl', 'gloo', etc.
    # system
    config["device"] = DEVICE # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    config["dtype"] = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


    # From train_shakespeare_char.py
    config["out_dir"] = 'out-shakespeare-char' if out_dir is None else out_dir
    config["eval_interval"] = 500 # keep frequent because we'll overfit
    config["eval_iters"] = 200
    config["log_interval"] = 10 # don't print too too often

    # we expect to overfit on this small dataset, so only save when val improves
    config["always_save_checkpoint"] = False

    config["wandb_log"] = False # override via command line if you like
    config["wandb_project"] = 'GW_LLM'
    config["wandb_run_name"] = 'mini-gpt'

    config["dataset"] = 'shakespeare_char'
    config["gradient_accumulation_steps"] = 1
    config["batch_size"] = 64
    config["block_size"] = 256 # context of up to 256 previous characters

    # baby GPT model :)
    config["n_layer"] = 6
    config["n_head"] = 6
    config["n_embd"] = 384
    config["dropout"] = 0.2

    config["learning_rate"] = 1e-3 # with baby networks can afford to go a bit higher
    config["max_iters"] = 5000
    config["lr_decay_iters"] = 5000 # make equal to max_iters usually
    config["min_lr"] = 1e-4 # learning_rate / 10 usually
    config["beta2"] = 0.99 # make a bit bigger because number of tokens per iter is small

    config["warmup_iters"] = 100 # not super necessary potentially

    # GW configs
    config["warmup_iters"] = 2000
    config["eval_iters"] = 200
    config["log_interval"] = 10
    config["block_size"] = 256
    config["batch_size"] = 64
    config["max_iters"] = 5000
    config["lr_decay_iters"] = 5000
    config["dropout"] = 0.2
    config["expert_k"] = 2

    # config["n_head"] = 6
    # config["n_embd"] = 384

    # Replace the default config values with the ones from the sweep
    config["n_layer"] = run_config["n_layer"] if "n_layer" in run_config else config["n_layer"]
    config["n_head"] = run_config["n_head"] if "n_head" in run_config else config["n_head"]
    config["n_embd"] = run_config["n_embd"] if "n_embd" in run_config else config["n_embd"]
    config["dropout"] = run_config["dropout"] if "dropout" in run_config else config["dropout"]
    config["batch_size"] = run_config["batch_size"] if "batch_size" in run_config else config["batch_size"]
    config["gradient_accumulation_steps"] = run_config["gradient_accumulation_steps"] if "gradient_accumulation_steps" in run_config else config["gradient_accumulation_steps"]
    if "learning_rate" in run_config:
        config["learning_rate"] = run_config["learning_rate"] 
        config["min_lr"] = config["learning_rate"] / 10
    config["max_iters"] = run_config["max_iters"] if "max_iters" in run_config else config["max_iters"]
    config["lr_decay_iters"] = run_config["lr_decay_iters"] if "lr_decay_iters" in run_config else config["max_iters"] # make equal to max_iters usually
    config["eval_iters"] = run_config["eval_iters"] if "eval_iters" in run_config else config["eval_iters"]
    config["eval_interval"] = run_config["eval_interval"] if "eval_interval" in run_config else config["eval_interval"]
    config["intermediate_outputs_training"] = run_config["intermediate_outputs_training"] if "intermediate_outputs_training" in run_config else False
    config["max_router_iter"] = config["n_layer"] // config["expert_k"] if 'max_router_iter' not in run_config else run_config["max_router_iter"]

    config["wandb_log"] = True
    config["dataset"] = 'babyLM_train_10M'
    
    config["model_cls"] = GPT if "Transformer" in run_config["condition"] else GW_GPT
    config["expansion_factor"] = 8 if "2xFF" in run_config["condition"] else 4
    config["n_embd"] = config["n_embd"] * 2 if "2n_embd" in run_config["condition"] else config["n_embd"]
    config["n_head"] = config["n_head"] * 2 if "2n_head" in run_config["condition"] else config["n_head"]
    config["identity_layers"] = "no_ID" not in run_config["condition"]
    config["always_save_checkpoint"] = True
    config["always_analyze_routing"] = True

        
    # if run_config["condition"] == "Big-Transformer":
    #     config["n_layer"] = run_config["n_layer"] * 2
    config["open_choice"] = "Ours" in run_config["condition"]
    config["weigh_experts"] = not run_config["condition"] == "Double"
    

    if run_config["condition"] == "Ours-2xT":
        config["max_router_iter"] = config["max_router_iter"] * 2


    # # fast debugging
    # torch.autograd.set_detect_anomaly(True)
    # max_router_iter = 2
    # max_iters = 2
    # batch_size = 2
    # block_size = 64



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
    model_args = dict(n_layer=config["n_layer"], max_router_iter=config["max_router_iter"], identity_layers=config["identity_layers"], intermediate_outputs_training=config["intermediate_outputs_training"], expert_k=config["expert_k"], open_choice=config["open_choice"], weigh_experts=config["weigh_experts"], n_head=config["n_head"], n_embd=config["n_embd"], block_size=config["block_size"],
                    bias=config["bias"], vocab_size=None, dropout=config["dropout"], expansion_factor=config["expansion_factor"])
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
    
    model.to(config["device"])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"], (config["beta1"], config["beta2"]), device_type)
    checkpoint = None # free up memory

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(analyze_routing=False, step=None):
        out = {}
        # done_whens = {}
        model.eval()

        for split in ['train', 'val']:
            routing_weights = []
            losses = {}
            # done_whens[split] = torch.zeros(eval_iters)
            for k in range(config["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    # logits, loss, done_when = model(X, Y)
                    if analyze_routing:
                        logits, loss, batch_routing_weights = model(X, Y, return_routing_weights=True)

                        routing_weights.extend(batch_routing_weights)
                    else:
                        logits, loss = model(X, Y)

                for key, value in loss.items():
                    if key not in losses:
                        losses[key] = torch.zeros(config["eval_iters"], requires_grad=False)
                    losses[key][k] = value.detach().item()

                # done_whens[split][k] = done_when
            out[split] = {key: value.mean().item() for key, value in losses.items()}
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
            # done_whens[split] = done_whens[split].mean()
            
        model.train()
        return out#, done_whens

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config["warmup_iters"]:
            return config["learning_rate"] * it / config["warmup_iters"]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config["lr_decay_iters"]:
            return config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        print("Starting iteration", iter_num)

        last_iteration = iter_num == config["max_iters"]

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if (iter_num % config["eval_interval"] == 0 and master_process) or last_iteration:
            # losses, done_whens = estimate_loss()
            analyze_routing = (last_iteration or config["always_analyze_routing"]) and config["model_cls"] == GW_GPT
            losses = estimate_loss(analyze_routing=analyze_routing, step=iter_num)
            print(f"step {iter_num}: train loss {losses['train']['total']:.4f}, val loss {losses['val']['total']:.4f}")
            if config["wandb_log"]:
                
                wandb_log = {
                    "iter": iter_num,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                }
                for split in ['train', 'val']:
                    for key, value in losses[split].items():
                        wandb_log[f"{split}/{key}"] = value
                wandb.log(wandb_log, step=iter_num+1)

            if losses['val']['total'] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses['val']['total']
                # done_when = done_whens['val']['total']
                if iter_num > 0 and save_checkpoint:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

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
            with ctx:
                # logits, loss, done_when = model(X, Y)
                logits, loss = model(X, Y)
                loss = loss['total']
                loss = loss / config["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * config["gradient_accumulation_steps"]
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(config["batch_size"] * config["gradient_accumulation_steps"], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:            
            break

    if ddp:
        destroy_process_group()
    


    return best_val_loss