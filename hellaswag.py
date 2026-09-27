"""Completion-style HellaSwag evaluation using a local validation JSONL file."""
import json
import torch
import torch.distributed as dist
from torch.nn import functional as F


def load_examples(path):
    # Read local data only; e.g. each row has one context, four endings, and label 0..3.
    with path.open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]
    if not examples:
        raise ValueError("HellaSwag validation file is empty")
    for example in examples:
        if not example["ctx"] or len(example["endings"]) != 4 or not 0 <= int(example["label"]) < 4:
            raise ValueError("Expected labeled HellaSwag validation examples with four endings")
    return examples


def render_example(example, encoder, block_size):
    context = encoder.encode(example["ctx"])
    endings = [encoder.encode(" " + ending) for ending in example["endings"]]
    if not context or any(not ending for ending in endings):
        raise ValueError("HellaSwag context and endings must contain tokens")
    width = len(context) + max(map(len, endings))
    if width > block_size:
        raise ValueError(f"HellaSwag example has {width} tokens, exceeding context {block_size}")
    tokens = torch.zeros((4, width), dtype=torch.long)
    mask = torch.zeros((4, width), dtype=torch.bool)
    for i, ending in enumerate(endings):
        row = context + ending
        tokens[i, :len(row)] = torch.tensor(row)
        # Score only the ending; e.g. two context tokens + three ending tokens -> 00111.
        mask[i, len(context):len(row)] = True
    return tokens, mask


def predict_endings(logits, tokens, mask):
    # Align next-token predictions; e.g. the last context token predicts the first ending token.
    predictions = logits[:, :-1, :].float()
    targets = tokens[:, 1:]
    losses = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1), reduction="none")
    losses = losses.view(tokens.size(0), -1)
    ending_mask = mask[:, 1:]
    summed = (losses * ending_mask).sum(dim=1)
    averaged = summed / ending_mask.sum(dim=1)
    return summed.argmin(), averaged.argmin()


@torch.no_grad()
def evaluate_hellaswag(model, examples, device, rank, world_size, ddp):
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    was_training = model.training
    model.eval()
    counts = torch.zeros(3, device=device, dtype=torch.long)
    try:
        # Use the unwrapped model: unequal example counts must not trigger DDP forward collectives.
        for index in range(rank, len(examples), world_size):
            example = examples[index]
            tokens, mask = render_example(example, encoder, model.config.block_size)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            predicted, normalized = predict_endings(logits, tokens, mask)
            label = int(example["label"])
            counts[0] += predicted == label
            counts[1] += normalized == label
            counts[2] += 1
        if ddp:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        correct, correct_norm, total = counts.tolist()
        return {"accuracy": correct / total, "accuracy_norm": correct_norm / total, "examples": total}
    finally:
        model.train(was_training)
