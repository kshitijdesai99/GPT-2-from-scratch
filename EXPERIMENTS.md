# Two-H100 training experiment — September 2026

## Setup

- Two H100 SXM 80 GB GPUs; 89 GB server disk; PyTorch 2.11.0 + CUDA 12.8.
- Current local source was copied to the server because GitHub cloning required authentication. Credentials were excluded.
- GPT-2-sized model: 12 layers, 12 heads, width 768, trained from random weights.
- FineWeb-Edu sample-10BT subset: 30,000,224 training tokens and 3,000,640 separate validation tokens, streamed on the server. The full 10B-token dataset was not downloaded.
- Batch: 64 sequences × 1,024 tokens × 2 GPUs × 4 accumulation steps = 524,288 tokens/update.
- BF16 training, compilation, gradient clipping, JSONL/TensorBoard metrics, validation every 100 updates, checkpoints every 25.

## Runs and observations

The first run lasted 10 minutes including startup/compilation, completing 724 updates. It saved step 700; the continuation restored that checkpoint and repeated the unsaved updates. A requested 30-minute continuation was stopped early after about 17 minutes because validation loss was worsening. Its last logged update was 2024; the final evaluated checkpoint was step 2000.

Typical training throughput was about 932,000 tokens/sec across both GPUs, with 97–98% utilization and roughly 49 GB memory used per GPU. Concurrent HellaSwag evaluation temporarily reduced throughput.

Validation loss fell from **10.9512** initially to **4.7873** at step 700, reached **4.3124** at step 1100, then rose to **4.9457** at step 2000 while training loss kept falling: evidence of overfitting the repeatedly reused subset.

CPU samples were captured approximately every minute using the same Taj Mahal question, seed 1337, temperature 0.8, top-k 50, and up to 48 new tokens. The first run's samples began around minute two. Responses progressed from punctuation to longer but inaccurate text; they did not reliably answer the question.

## HellaSwag

All 10,042 validation examples from `Rowan/hellaswag` were scored using the same BF16, completion-style evaluator. Only ending tokens count. Normalized scoring divides each ending's loss by its token count.

| Model | Accuracy | Normalized accuracy |
|---|---:|---:|
| Step 1000 | 25.83% | 25.30% |
| Step 2000 | 26.39% | 25.75% |
| Original pretrained GPT-2 | 28.63% | 29.65% |
| Random guessing (expected) | 25% | 25% |

The normalized gain was only 0.45 percentage points (45 more correct answers). These results do not demonstrate a reliable generalization improvement. No initial random-model HellaSwag baseline was measured.

A separate identical 8,192-token validation comparison gave loss **4.8876** for step 700 versus **3.3623** for original GPT-2. This smaller batch differs from regular training validation above.

## Conclusion and cleanup

Distributed training, checkpoint restoration, logging, and evaluation worked on the tested server. More fresh training data is the next useful step; filling GPU memory or repeating this small subset longer is not sufficient.

Training, sampling, and monitoring were stopped. Only text results/logs were retained locally; the model-weight download was cancelled and its partial copy removed at the user's request. No weights, datasets, credentials, or generated logs are committed. Instance deletion was left to the user.
