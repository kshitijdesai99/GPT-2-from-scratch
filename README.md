- **Coding style:** Use concise, consistent comments explaining each operation with a concrete value or tensor-shape example that matches the model configuration; split chained operations into clear steps.


## Where to find each part

Start with `train_gpt2.py:main()` to follow a training run from setup to evaluation.
The modules are ordinary Python files; there is no trainer framework to learn.

| File | Responsibility / edit here when... |
|---|---|
| `train_gpt2.py` | Run settings, device/DDP setup, resume, learning-rate schedule, gradient clipping, and the training loop. |
| `model.py` | Attention, MLP, transformer blocks, GPT configuration, forward pass, pretrained weights, and optimizer groups. |
| `data_loader.py` | Read local shards, form shifted input/target batches, distribute batches across GPUs, and reset the data position. |
| `evaluation.py` | Compute validation loss over held-out batches. |
| `hellaswag.py` | Load local benchmark examples and score their four possible endings. |
| `checkpoint.py` | Save model/optimizer/data/RNG state safely across ranks; resume orchestration stays in the training entry point. |
| `tracking.py` | Write JSONL metrics and TensorBoard curves, including resume handling. |
| `fineweb.py` | Download, tokenize, and write the dataset shards before training. |

For experimenting with the model alone, use `from model import GPT, GPTConfig`.
Existing `python train_gpt2.py` and `torchrun ... train_gpt2.py` commands still work.

## FineWeb training on one server with eight GPUs

Prepare data once with `python fineweb.py` on the server where downloading is intended.
Do not launch preprocessing with `torchrun`. It writes raw uint16 shards into
`edu_fineweb10B/`: shard 0 is validation, and the other shards are training data.
Budget roughly 20 GB for the token files plus the downloaded Hugging Face dataset/cache.
Training itself reads local shards and does not download the dataset.

Use a CUDA-enabled PyTorch environment with NCCL and eight visible GPUs. Install
this project's dependencies in that server environment; do not copy the Mac `.venv`.
First run a short check (this still uses the full model and production microbatch):

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py --run-steps 2 --val-batches 2 --out-dir checkpoints-smoke
```

Then start the full run in its own directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py --out-dir checkpoints
```

Defaults: 64 sequences/GPU, 1024 tokens/sequence, 524288 global tokens/update,
19073 updates, and 715 warmup updates. Eight GPUs need one microbatch per update.
If the server runs out of memory, reduce `--batch-size` (for example to 16); gradient
accumulation preserves the global batch size. Choose a fresh output directory when
changing the batch layout. Actual CUDA memory usage and throughput require a server test.

Validation runs initially, every 100 updates, and at the end, using 20 distributed
validation batches. `--val-batches` and `--val-every` control these settings. Training
writes `latest.pt` atomically every 250 updates and at the end; `--save-every` changes
the interval. This keeps the latest checkpoint rather than one file per save.

Resume from the last completed save:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py --out-dir checkpoints --resume checkpoints/latest.pt
```

Resume restores weights, optimizer, data position, per-rank RNG state, and the next
schedule step. Keep the same GPU count, microbatch, sequence length, global batch,
learning-rate settings, and total schedule length. Shards must remain unchanged;
the loader checks their names and byte sizes. All ranks must see the same shards and
resume file. `--run-steps` limits additional updates without changing the full LR
schedule, so a smoke-test checkpoint can also be continued with matching settings.
`--no-compile` disables model and clipping compilation for troubleshooting.


## Metrics and HellaSwag

Rank 0 writes `metrics.jsonl` and TensorBoard events under `--out-dir`.
Training records loss, learning rate, gradient norm, step milliseconds, and global
training tokens/sec every update. Validation records mean loss every 100 updates.
All metric steps count completed optimizer updates. Resume removes JSONL records
and hides TensorBoard events newer than the restored checkpoint; initial evaluation
can produce another result at the resume step.

Install the updated project dependencies with `uv sync` on the training server, then view the curves:

```bash
tensorboard --logdir checkpoints/tensorboard
```

HellaSwag is opt-in and requires the official labeled
[validation JSONL](https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl)
already stored on the training server. Training never downloads this file.
Enable it on both the initial run and resume:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py --out-dir checkpoints --hellaswag-file /path/to/hellaswag_val.jsonl
```

The full supplied file is evaluated at startup, every 250 updates, and at the end;
`--hellaswag-every` changes the interval. Each GPU scores different examples, and
correct/total counts are summed across GPUs. The evaluator uses the uncompiled,
unwrapped model to support variable lengths and unequal example counts per rank.
It logs both summed ending likelihood accuracy (`accuracy`) and mean per-token
ending likelihood accuracy (`accuracy_norm`), excluding context and padding.
This follows [Karpathy's completion-style scoring](https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py),
which is not directly interchangeable with other evaluation harnesses.
Omit `--hellaswag-file` for a training smoke test without benchmark evaluation.
