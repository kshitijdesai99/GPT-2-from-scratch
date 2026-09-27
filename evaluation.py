"""Measure next-token validation loss without updating model weights.

HellaSwag multiple-choice scoring lives separately in hellaswag.py.
"""
import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate(model, loader, device, batches, ddp, world_size):
    # Restart validation so evaluations use the same held-out batches.
    loader.reset()
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    try:
        for _ in range(batches):
            x, y = loader.next_batch()
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                _, loss = model(x.to(device), y.to(device))
            total_loss += loss.detach() / batches
        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= world_size
        return total_loss.item()
    finally:
        model.train(was_training)
