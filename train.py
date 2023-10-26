"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import cProfile
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import polars as pl
import argparse
import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task, select_batches_sorted_by_column
from export import model_export

# -----------------------------------------------------------------------------


def setup_args_and_run(**override_args):
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Training settings")

    # General
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory.")
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--always_save_checkpoint", action="store_true", default=True)
    parser.add_argument(
        "--init_from", type=str, choices=["scratch", "resume"], default="scratch"
    )
    parser.add_argument(
        "--config_override",
        type=str,
        default=None,
        help="Path to the configuration override file.",
    )

    # wandb logging
    parser.add_argument("--wandb_log", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="llamac")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )

    # data
    parser.add_argument("--batch_size", type=int, default=32)  # 128
    parser.add_argument("--max_seq_len", type=int, default=128)  # 256
    parser.add_argument(
        "--vocab_source", type=str, choices=["llama2", "custom"], default="llama2"
    )
    parser.add_argument("--vocab_size", type=int, default=32000)

    # batch selection
    parser.add_argument(
        "--batch_selection",
        type=str,
        choices=["random", "sort_column"],
        default="sort_column",
        help="How to select batches from the dataset",
    )
    parser.add_argument(
        "--sort_by_column",
        type=str,
        default="flesch_kincaid_grade",
        help="Column name to sort by if batch_selection is set to sort_column",
    )
    parser.add_argument(
        "--sort_by_direction",
        type=str,
        default="desc",
        choices=["asc", "desc"],
        help="Sort direction if batch_selection is set to sort_column",
    )

    # model
    parser.add_argument("--dim", type=int, default=288)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_kv_heads", type=int, default=6)
    parser.add_argument("--multiple_of", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--iter_num", type=int, default=0, help="Initial iteration number"
    )
    parser.add_argument(
        "--best_val_loss", type=float, default=1e9, help="Initial best validation loss"
    )

    # adamw optimizer
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # learning rate decay settings
    parser.add_argument("--decay_lr", action="store_true", default=True)
    parser.add_argument("--warmup_iters", type=int, default=1000)

    # system
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--master_process", type=bool, default=False)

    args = parser.parse_args()

    # -----------------------------------------------------------------------------

    # Override the arguments if provided in the function call
    for key, value in override_args.items():
        print(f"Overriding {key} with {value}")
        setattr(args, key, value)

    # Call the main function with the modified args
    main(args)


# -----------------------------------------------------------------------------


def main(args):
    # Validate settings
    assert args.vocab_source in ["llama2", "custom"]
    assert (
        args.vocab_source == "custom" or args.vocab_size == 32000
    ), "The vocab from Meta has 32K tokens"

    # -----------------------------------------------------------------------------

    # Initialize distributed training settings if applicable
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        args.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(args.device)
        args.master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        args.master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = (
        args.gradient_accumulation_steps
        * ddp_world_size
        * args.batch_size
        * args.max_seq_len
    )
    if args.master_process:
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")
        print(
            f"Breaks down as: {args.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {args.batch_size} batch size * {args.max_seq_len} max seq len"
        )

    if args.master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------------

    # Set random seed and device type
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # -----------------------------------------------------------------------------
    # Batch selection
    def get_select_func(selection_strategy, sort_column=None, sort_direction="asc"):
        if selection_strategy == "sort_column" and sort_column:
            ascending = True if sort_direction == "asc" else False
            return partial(
                select_batches_sorted_by_column,
                column_name=sort_column,
                ascending=ascending, 
            )
        else:
            return None  # default

    select_function = get_select_func(
        args.batch_selection, args.sort_by_column, args.sort_by_direction
    )

    # -----------------------------------------------------------------------------

    iter_batches = partial(
        Task.iter_batches,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        vocab_source=args.vocab_source,
        device=args.device,
        num_workers=0)

    # -----------------------------------------------------------------------------

    # Initialize the model
    model_args = dict(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    if args.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif args.init_from == "resume":
        print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in [
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "vocab_size",
            "multiple_of",
            "max_seq_len",
        ]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    model.to(args.device)
    print(next(model.parameters()).device)  # Should print something like cuda:0

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # -----------------------------------------------------------------------------

    # optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if args.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # -----------------------------------------------------------------------------

    # compile the model
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
        print("compilation done")

    # -----------------------------------------------------------------------------

    # wrap model into DDP container
    if ddp:
        print("wrapping model into DDP container")
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    # -----------------------------------------------------------------------------

    # logging
    if args.wandb_log and args.master_process:
        import wandb

        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name
        )  # , config=config)

    # -----------------------------------------------------------------------------

    train_model(
        model,
        optimizer,
        iter_batches,
        args,
        ctx,
        ddp,
        tokens_per_iter,
        scaler,
        model_args,
    )

    if ddp:
        destroy_process_group()


# training loop
def train_model(
    model,
    optimizer,
    iter_batches,
    args,
    ctx,
    ddp,
    tokens_per_iter,
    scaler,
    model_args,
):
    # args
    train_iter_batches = partial(
        iter_batches,
    )
    out_dir = args.out_dir
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    eval_only = args.eval_only
    always_save_checkpoint = args.always_save_checkpoint
    wandb_log = args.wandb_log
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    grad_clip = args.grad_clip
    max_iters = args.max_iters
    decay_lr = args.decay_lr
    best_val_loss = args.best_val_loss
    iter_num = args.iter_num

    print(f"Model is on: {args.device}")

    # functions

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(args.eval_iters)  # keep on CPU
            for k in range(args.eval_iters):
                X, Y, global_ix, metadata = next(batch_iter)
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.max_iters:
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    batch_indices_trained = []
    train_batch_iter = train_iter_batches(split="train")

    # Fetch the very first batch and its indices
    X, Y, global_ix, metadata = next(train_batch_iter)
    batch_indices_trained.append(global_ix)
    t0 = time.time()

    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets andf write checkpoints
        if iter_num % eval_interval == 0 and args.master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if args.wandb_log:
                import wandb

                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        # "config": config,
                        "batch_indices_trained": batch_indices_trained,
                    }

                    # Save attention weights
                    checkpoint[
                        "last_attn_weights"
                    ] = raw_model.get_all_layers_attention_weights()

                    # Save token embeddings
                    tok_embeddings = raw_model.tok_embeddings.weight.data.clone()
                    checkpoint["tok_embeddings"] = tok_embeddings

                    # save paths
                    checkpoint_filename = f"ckpt/ckpt_{iter_num}.pt"
                    print(
                        f"saving checkpoint to {os.path.join(out_dir, checkpoint_filename)}"
                    )
                    torch.save(checkpoint, os.path.join(out_dir, checkpoint_filename))

                    model_filename = f"models/model_{iter_num}.bin"
                    print(f"saving model to {os.path.join(out_dir, model_filename)}")
                    model_export(
                        raw_model, os.path.join(out_dir, model_filename), version=0
                    )

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
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, global_ix, metadata = next(train_batch_iter)
            batch_indices_trained.append(global_ix)  # append the batch indices here
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
        if iter_num % log_interval == 0 and args.master_process:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )

        # Reset the attention weights after each iteration
        # raw_model.reset_all_layers_attention_weights() # debug check this

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            print(f"reached max_iters {max_iters}, terminating training")
            break

    return


if __name__ == "__main__":
    setup_args_and_run()
    # cProfile.run('setup_args_and_run()', 'output_profile_stats')