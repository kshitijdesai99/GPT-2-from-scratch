"""Training entry point: settings, device setup, resume, and optimizer updates.

Start with main() to follow the run. See README.md for the file responsibility map.
"""
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from data_loader import DataLoaderLite
from evaluation import evaluate
from checkpoint import save_checkpoint
from tracking import MetricsLogger
from hellaswag import load_examples, evaluate_hellaswag


def synchronize_device(device):
    # Wait for queued accelerator work before timing; e.g. Apple GPU work uses MPS.
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
    # CPU operations already complete synchronously, so no wait is needed.

@torch.compile
def clip_gradients(grads):
    # combine gradient norms into one global norm
    norms = torch.stack([torch.linalg.vector_norm(g) for g in grads])
    total_norm = torch.linalg.vector_norm(norms)

    # Limit the norm to 1; e.g. norm = 2 gives scale = 0.5
    scale = torch.clamp(1.0/(total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(scale)
    return total_norm


def get_lr(step, args):
    # Warm up to max_lr, then decay across the full run even during a short smoke test.
    if step < args.warmup_steps:
        return args.max_lr * (step + 1) / args.warmup_steps
    ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    ratio = min(max(ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    min_lr = args.max_lr * 0.1
    return min_lr + coefficient * (args.max_lr - min_lr)


def parse_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train GPT-2 on raw FineWeb-Edu uint16 shards")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "edu_fineweb10B")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "checkpoints")
    parser.add_argument("--resume", type=Path, help="Path to a checkpoint written by this script")
    parser.add_argument("--batch-size", type=int, default=64, help="Sequences per GPU")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--total-batch-size", type=int, default=524288, help="Tokens per global optimizer step")
    parser.add_argument("--max-steps", type=int, default=19073, help="Total schedule length, including resumed steps")
    parser.add_argument("--warmup-steps", type=int, default=715)
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--hellaswag-file", type=Path, help="Local labeled validation JSONL; omitted disables HellaSwag")
    parser.add_argument("--hellaswag-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--run-steps", type=int, help="Stop and checkpoint after this many updates without shortening the LR schedule")
    parser.add_argument("--no-compile", action="store_true", help="Disable compilation for troubleshooting")
    args = parser.parse_args()
    for name in ("batch_size", "sequence_length", "total_batch_size", "max_steps", "val_every", "val_batches", "save_every", "hellaswag_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup_steps < args.max_steps:
        parser.error("warmup steps must be between 0 and max_steps - 1")
    if not math.isfinite(args.max_lr) or args.max_lr <= 0:
        parser.error("max_lr must be finite and positive")
    if args.sequence_length > GPTConfig.block_size:
        parser.error("sequence length exceeds the model context window")
    if args.run_steps is not None and args.run_steps <= 0:
        parser.error("--run-steps must be positive")
    return args


def main():
    import os
    import time

    # 1. Read settings and assign one process to each GPU.
    args = parse_args()
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("The torchrun training path requires CUDA/NCCL")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = torch.device(device).type
    master_process = rank == 0
    logger = None

    try:
        # 2. Create data loaders and check the global batch size.
        global_microbatch = args.batch_size * args.sequence_length * world_size
        if args.total_batch_size % global_microbatch:
            raise ValueError("total_batch_size must be divisible by batch_size * sequence_length * world_size")
        grad_accum_steps = args.total_batch_size // global_microbatch
        # Resume requires the same batch layout/schedule to preserve data order and LR progression.
        run_config = {name: getattr(args, name) for name in (
            "batch_size", "sequence_length", "total_batch_size", "max_steps", "warmup_steps", "max_lr"
        )}
        run_config.update(world_size=world_size, device_type=device_type)
        if master_process:
            print(f"device: {device}; world_size: {world_size}; accumulation: {grad_accum_steps}", flush=True)
        train_loader = DataLoaderLite(args.batch_size, args.sequence_length, rank, world_size, args.data_dir, "train")
        val_loader = DataLoaderLite(args.batch_size, args.sequence_length, rank, world_size, args.data_dir, "val")

        hellaswag_examples = load_examples(args.hellaswag_file) if args.hellaswag_file else None

        # 3. Build the model and restore a checkpoint when resuming.
        torch.manual_seed(1337)
        torch.set_float32_matmul_precision("high")
        checkpoint = None
        start_step = 0
        config = GPTConfig(vocab_size=50304)
        if args.resume:
            # Load only tensors and basic containers; no arbitrary checkpoint objects are needed.
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            if checkpoint.get("version") != 1 or checkpoint["run_config"] != run_config:
                raise ValueError("Checkpoint version or training batch/schedule/device settings do not match")
            if checkpoint["model_config"] != vars(config):
                raise ValueError("Checkpoint model configuration does not match")
            start_step = checkpoint["step"]
            if not 0 <= start_step <= args.max_steps or len(checkpoint["rng_states"]) != world_size:
                raise ValueError("Checkpoint has an invalid step or rank count")
            train_loader.load_state_dict(checkpoint["loader"])
        elif (args.out_dir / "latest.pt").exists():
            raise FileExistsError("Output checkpoint already exists; use --resume or a different --out-dir")

        if master_process:
            logger = MetricsLogger(args.out_dir, start_step if args.resume else None)

        raw_model = GPT(config).to(device).train()
        if checkpoint:
            raw_model.load_state_dict(checkpoint["model"])
        optimizer = raw_model.configure_optimizers(0.1, args.max_lr, device, master_process)
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model
        if not args.no_compile:
            model = torch.compile(model)
        # Use the tested compiled clipping path unless compilation is explicitly disabled.
        clip = clip_gradients.__wrapped__ if args.no_compile else clip_gradients
        if checkpoint:
            rng = checkpoint["rng_states"][rank]
            torch.set_rng_state(rng["cpu"])
            if device_type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], device)
            elif device_type == "mps":
                torch.mps.set_rng_state(rng["mps"])
            del checkpoint

        # 4. Evaluate before training to record the starting performance.
        stop_step = min(args.max_steps, start_step + args.run_steps) if args.run_steps else args.max_steps
        val_loss = evaluate(model, val_loader, device, args.val_batches, ddp, world_size)
        if master_process:
            print(f"validation step {start_step}: loss {val_loss:.4f}", flush=True)
            logger.log(start_step, "val", loss=val_loss)
        if hellaswag_examples is not None:
            scores = evaluate_hellaswag(raw_model, hellaswag_examples, device, rank, world_size, ddp)
            if master_process:
                print(f"hellaswag step {start_step}: {scores}", flush=True)
                logger.log(start_step, "hellaswag", **scores)
        # 5. Accumulate gradients, update weights, then save/evaluate when due.
        for step in range(start_step, stop_step):
            synchronize_device(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                sync = model.no_sync() if ddp and micro_step < grad_accum_steps - 1 else nullcontext()
                with sync:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x.to(device), y.to(device))
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach()
                    loss.backward()
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
                loss_accum /= world_size
            with torch.no_grad():
                norm = clip([p.grad for p in raw_model.parameters() if p.grad is not None])
            # Check the synchronized loss/gradient norm before corrupting parameters with NaNs.
            if not torch.isfinite(loss_accum).item() or not torch.isfinite(norm).item():
                raise FloatingPointError(f"Non-finite loss or gradient norm at step {step}")
            lr = get_lr(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            synchronize_device(device)
            elapsed = torch.tensor(time.perf_counter() - t0, device=device, dtype=torch.float32)
            if ddp:
                # Global throughput uses the slowest rank, not just rank 0's local duration.
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if master_process:
                seconds = elapsed.item()
                print(f"step {step + 1}, loss: {loss_accum.item():.4f}, lr: {lr:.6g}, norm: {norm.item():.4f}, "
                      f"dt: {seconds * 1000:.2f}ms, tok/sec: {args.total_batch_size / seconds:.2f}", flush=True)
                logger.log(step + 1, "train", loss=loss_accum.item(), lr=lr, grad_norm=norm.item(),
                           step_ms=seconds * 1000, tokens_per_sec=args.total_batch_size / seconds)
            completed = step + 1
            if completed % args.save_every == 0 or completed == stop_step:
                save_checkpoint(args.out_dir / "latest.pt", raw_model, optimizer, train_loader,
                                completed, run_config, device, ddp, rank, world_size)
            if completed % args.val_every == 0 or completed == stop_step:
                val_loss = evaluate(model, val_loader, device, args.val_batches, ddp, world_size)
                if master_process:
                    print(f"validation step {completed}: loss {val_loss:.4f}", flush=True)
                    logger.log(completed, "val", loss=val_loss)
            if hellaswag_examples is not None and (completed % args.hellaswag_every == 0 or completed == stop_step):
                scores = evaluate_hellaswag(raw_model, hellaswag_examples, device, rank, world_size, ddp)
                if master_process:
                    print(f"hellaswag step {completed}: {scores}", flush=True)
                    logger.log(completed, "hellaswag", **scores)
    finally:
        try:
            if logger is not None:
                logger.close()
        finally:
            if ddp and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
