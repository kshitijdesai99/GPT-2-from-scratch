"""Save model, optimizer, data position, and per-rank random states together.

The training entry point loads this state when --resume is supplied.
"""
import torch
import torch.distributed as dist


def save_checkpoint(path, model, optimizer, loader, step, run_config, device, ddp, rank, world_size):
    import os
    from dataclasses import asdict
    from pathlib import Path

    # Preserve each rank's random state; model/optimizer states are synchronized by DDP.
    rng = {"cpu": torch.get_rng_state()}
    device_type = torch.device(device).type
    if device_type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state(device)
    elif device_type == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    rng_states = [None] * world_size if ddp else [rng]
    if ddp:
        dist.all_gather_object(rng_states, rng)
    error = [None]
    if rank == 0:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "version": 1, "step": step, "model_config": asdict(model.config),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "loader": loader.state_dict(), "run_config": run_config, "rng_states": rng_states,
            }
            # Replace the last checkpoint only after the new file is completely written.
            temporary = path.with_name(path.name + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
            print(f"saved {path} at step {step}", flush=True)
        except Exception as exc:
            error[0] = f"Checkpoint save failed: {exc}"
    if ddp:
        # All ranks wait for the save and receive any disk error instead of training ahead.
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
