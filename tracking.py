"""Rank-zero JSONL metrics and TensorBoard curves."""
import json
import time


class MetricsLogger:
    def __init__(self, out_dir, resume_step=None):
        from torch.utils.tensorboard import SummaryWriter

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metrics.jsonl"
        # Remove updates lost since the checkpoint; e.g. resume at 250 drops step 251.
        if resume_step is not None and path.exists():
            records = []
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A crash can leave an unfinished final line.
                if record["step"] <= resume_step:
                    records.append(json.dumps(record) + "\n")
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(records))
            temporary.replace(path)
        elif path.exists():
            raise FileExistsError("Metrics already exist; use --resume or a fresh --out-dir")
        self.writer = SummaryWriter(
            log_dir=str(out_dir / "tensorboard"),
            purge_step=resume_step + 1 if resume_step is not None else None,
        )
        self.file = path.open("a", encoding="utf-8")

    def log(self, step, split, **metrics):
        # Use completed updates on every curve; e.g. step 100 means 100 optimizer steps.
        record = dict(step=step, split=split, time=time.time(), **metrics)
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()
        for name, value in metrics.items():
            self.writer.add_scalar(f"{split}/{name}", value, step)
        self.writer.flush()

    def close(self):
        try:
            self.writer.close()
        finally:
            self.file.close()
