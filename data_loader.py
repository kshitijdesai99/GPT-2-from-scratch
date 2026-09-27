"""Read local FineWeb shards and build shifted training or validation batches.

Example: tokens [10, 20, 30, 40] give x=[10, 20, 30], y=[20, 30, 40].
"""
import torch


class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, data_dir=None, split="train"):
        import bisect
        from pathlib import Path
        import numpy as np

        if B <= 0 or T <= 0 or not 0 <= process_rank < num_processes:
            raise ValueError("Require positive B/T and a rank within num_processes")
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.stride = B * T * num_processes
        root = Path(data_dir) if data_dir else Path(__file__).parent / "edu_fineweb10B"
        # Match the raw files produced by fineweb.py; keep validation separate from training.
        self.shards = [path for path in sorted(root.glob(f"edufineweb_{split}_*"))
                       if path.name.rsplit("_", 1)[-1].isdigit()]
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards found in {root}; prepare fineweb.py output first")
        self.offsets = [0]
        self.manifest = []
        for path in self.shards:
            size = path.stat().st_size
            if not path.is_file() or size == 0 or size % 2:
                raise ValueError(f"Invalid raw uint16 shard: {path}")
            self.manifest.append((path.name, size))
            self.offsets.append(self.offsets[-1] + size // 2)
        self.total_tokens = self.offsets[-1]
        if self.total_tokens < self.stride + 1:
            raise ValueError(f"{split} requires at least {self.stride + 1} tokens across its shards")
        self.current_position = 0
        self.epoch = 0
        self._cached_index = None
        self._cached_tokens = None
        self._np = np
        self._bisect = bisect
        if process_rank == 0:
            print(f"{split}: {len(self.shards)} shards, {self.total_tokens:,} tokens", flush=True)

    def reset(self):
        self.current_position = 0
        self.epoch = 0

    def state_dict(self):
        # Every rank shares this round position; its rank-specific offset is added during reads.
        return {"position": self.current_position, "epoch": self.epoch, "manifest": self.manifest}

    def load_state_dict(self, state):
        if state["manifest"] != self.manifest:
            raise ValueError("Training shard names/sizes changed since the checkpoint")
        position = state["position"]
        if position < 0 or position % self.stride or position + self.stride + 1 > self.total_tokens:
            raise ValueError("Checkpoint contains an invalid data position")
        self.current_position = position
        self.epoch = state["epoch"]

    def next_batch(self):
        start = self.current_position + self.process_rank * self.B * self.T
        count = self.B * self.T + 1
        # Read only this microbatch; e.g. 65537 IDs become int64 without loading the full dataset.
        buf = self._np.empty(count, dtype=self._np.int64)
        copied = 0
        while copied < count:
            index = self._bisect.bisect_right(self.offsets, start) - 1
            if self._cached_index != index:
                self._cached_tokens = self._np.memmap(self.shards[index], dtype=self._np.uint16, mode="r")
                self._cached_index = index
            offset = start - self.offsets[index]
            length = min(count - copied, self.offsets[index + 1] - start)
            # Continue across shard boundaries, including a short final shard.
            buf[copied:copied + length] = self._cached_tokens[offset:offset + length]
            start += length
            copied += length
        if buf.max() >= 50257:
            raise ValueError("Shard contains a token ID outside the GPT-2 tokenizer vocabulary")
        tokens = torch.from_numpy(buf)
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        self.current_position += self.stride
        # All ranks reset together; drop only the final incomplete distributed round.
        if self.current_position + self.stride + 1 > self.total_tokens:
            self.current_position = 0
            self.epoch += 1
        return x, y
