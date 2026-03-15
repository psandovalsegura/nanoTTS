# -----------------------------------------------------------------------------
# From train.py of nanoGPT: https://github.com/karpathy/nanoGPT
# Changes are marked by @psando, and described below:
# - replaced poor man's data loader with libritts
# -----------------------------------------------------------------------------

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
import random                 # @psando
from contextlib import nullcontext

import torch
import torchaudio # @psando
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from libritts_dataset import TTSDataset, create_collate_fn # @psando
from decoder.pretrained import WavTokenizer                # @psando 

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = '/fs/nexus-scratch/psando/nanotts'
eval_interval = 100
log_interval = 10
eval_iters = 4
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'nanotts'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset_root = '/fs/nexus-scratch/psando' # @psando
dataset_url = 'train-clean-100'           # @psando
bpe_tokenizer_path = 'libritts_bpe.json'  # @psando
gradient_accumulation_steps = 1 # TODO: @psando: tune after shard dataset, 4 #5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 2048                         # @psando
# @psando: wavtokenizer
wavtokenizer_dir = '/fs/nexus-scratch/psando/WavTokenizer'
wavtokenizer_config = 'wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
wavtokenizer_ckpt = 'WavTokenizer_small_600_24k_4096.ckpt'
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 0.001 #6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # TODO: @psando: tune after shard ds, 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster # @psando: error if True, must fix
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    out_dir = os.path.join(out_dir, wandb_run_name)
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# @psando: add libritts dataset and wavtokenizer
libritts_dataset = torchaudio.datasets.LIBRITTS(
    root=dataset_root,
    url=dataset_url,
    download=False,
)

wt_device = torch.device('cpu')
config_path = os.path.join(wavtokenizer_dir, wavtokenizer_config)
model_path = os.path.join(wavtokenizer_dir, wavtokenizer_ckpt)
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(wt_device)
bandwidth_id = torch.tensor([0], device=wt_device)
if not hasattr(wavtokenizer, 'bandwidth_id'):
    wavtokenizer.bandwidth_id = bandwidth_id

tts_dataset = TTSDataset(
    libritts_dataset=libritts_dataset,
    bpe_tokenizer_path=bpe_tokenizer_path,
    wav_tokenizer=wavtokenizer
)

dataloader = torch.utils.data.DataLoader(
    tts_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    collate_fn=create_collate_fn(tts_dataset.pad_id)
)

dataloader_iter = iter(dataloader)
dataloader_iter_estimate = iter(dataloader)

# @psando: validation will just choose one of a few prompts, and a random speaker id
val_prompts = ['My name is Pedro Sandoval Segura.', 'Nano text-to-speech was created at University of Maryland.']
val_speaker_ids = [19, 26, 27]

def get_batch(split):
    global dataloader_iter
    global dataloader_iter_estimate
    if split == 'train':
        try:
            batch = next(dataloader_iter)            
        except StopIteration:
            dataloader_iter = iter(dataloader) # @psando: new epoch
            batch = next(dataloader_iter)
        batch = tuple(b.pin_memory().to(device, non_blocking=True) for b in batch)
    elif split == 'estimate_train':
        try:
            batch = next(dataloader_iter_estimate)            
        except StopIteration:
            dataloader_iter_estimate = iter(dataloader) # @psando: new epoch
            batch = next(dataloader_iter_estimate)
        batch = tuple(b.pin_memory().to(device, non_blocking=True) for b in batch)
    return batch

def get_val_batch():
    normalized_text = random.choice(val_prompts)
    speaker_id = random.choice(val_speaker_ids)
    prompt_string = f"<BOS>{normalized_text}<SPK_{speaker_id}><AUDIO_START>"
    text_ids = tts_dataset.text_tokenizer.encode(prompt_string).ids 
    caption = f'spk_{speaker_id}_{normalized_text.replace(" ", "_")}'
    batch = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(device)
    return batch, caption

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")     
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# @psando: removed the option to initialize from pretrained gpt2
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# @psando: validate performance through generation
@torch.no_grad()
def estimate_val():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('estimate_train')
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    out['estimate_train'] = losses.mean()

    # generate an audio sample
    val, caption = get_val_batch()
    val_cont = model.generate(val, max_new_tokens=500, temperature=1.0, top_k=None)
    out['gen_audio'] = tts_dataset._get_waveform_from_generated_sequence(val_cont.cpu())
    out['gen_caption'] = caption
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        out = estimate_val()
        print(f"step {iter_num}: train loss {out['estimate_train']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train_loss": out['estimate_train'],
                "lr": lr,
                "gen_audio": wandb.Audio(out['gen_audio'].numpy().T, sample_rate=24000, caption=out['gen_caption']),
                "mfu": running_mfu*100, # convert to percentage
            })
        if always_save_checkpoint:
            # best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'train_loss': out['estimate_train'],
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num:06d}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
