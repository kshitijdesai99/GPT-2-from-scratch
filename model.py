"""GPT-2 architecture, pretrained weights, and optimizer parameter groups.

Read in order: attention -> MLP -> transformer block -> full GPT model.
"""
from dataclasses import dataclass
import inspect
import torch
from torch import nn
from torch.nn import functional as F

# Shape examples use B=2 sequences, T=5 tokens, C=768 features, H=12 heads, D=64 features/head.
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Require equal-sized heads; e.g. 768 features / 12 heads = 64 features/head.
        assert config.n_embd % config.n_head == 0

        # Learn query, key and value projections together; e.g. 768 -> 2304 features.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Learn how to mix the joined heads; e.g. 768 -> 768 features.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Save head count and embedding width; e.g. H=12 and C=768.
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # Read batch size, token count and feature width; e.g. (2, 5, 768).
        B, T, C = x.size()

        # Project to Q, K and V: (B, T, C) -> (B, T, 3C); e.g. (2, 5, 768) -> (2, 5, 2304).
        qkv = self.c_attn(x)
        # Split into three C-wide tensors; e.g. Q, K and V each have shape (2, 5, 768).
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split each tensor into heads: (B, T, C) -> (B, T, H, D); e.g. (2, 5, 12, 64).
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        # Move heads before tokens: (B, T, H, D) -> (B, H, T, D); e.g. (2, 12, 5, 64).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply causal scaled dot-product attention; e.g. (2, 12, 5, 64) -> (2, 12, 5, 64).
        # Each token uses only current/past tokens; PyTorch selects the attention backend.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Move tokens before heads: (B, H, T, D) -> (B, T, H, D); e.g. (2, 5, 12, 64).
        y = y.transpose(1, 2)
        # Copy if needed to make storage contiguous for view; shape stays (2, 5, 12, 64).
        y = y.contiguous()
        # Merge heads: (B, T, H, D) -> (B, T, C); e.g. (2, 5, 12, 64) -> (2, 5, 768).
        y = y.view(B, T, C)
        # Mix head outputs with a learned projection; e.g. (2, 5, 768) -> (2, 5, 768).
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Learn an expansion to 4C features; e.g. 768 -> 3072.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Use the tanh approximation of GELU; e.g. -1 -> about -0.159, 1 -> about 0.841.
        self.gelu = nn.GELU(approximate="tanh")
        # Learn a projection back to C features; e.g. 3072 -> 768.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # Expand each token's features: (B, T, C) -> (B, T, 4C); e.g. (2, 5, 3072).
        x = self.c_fc(x)
        # Apply GELU elementwise; e.g. -1 -> about -0.159; shape stays (2, 5, 3072).
        x = self.gelu(x)
        # Project back to the embedding width: (B, T, 4C) -> (B, T, C); e.g. (2, 5, 768).
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Normalize each token over C features before attention; e.g. 768 features/token.
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Let tokens attend to current/past tokens; e.g. token 2 can use tokens 0, 1 and 2.
        self.attn = CausalSelfAttention(config)
        # Normalize each token over C features before the MLP; e.g. 768 features/token.
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Transform each token independently; e.g. 768 -> 3072 -> 768 features.
        self.mlp = MLP(config)

    def forward(self, x):
        # Normalize token features; e.g. (2, 5, 768) -> (2, 5, 768).
        normalized = self.ln_1(x)
        # Compute attention updates; e.g. (2, 5, 768) -> (2, 5, 768).
        update = self.attn(normalized)
        # Add the attention residual elementwise; e.g. original 0.5 + update 0.2 = 0.7.
        x = x + update
        # Normalize the updated token features; e.g. (2, 5, 768) -> (2, 5, 768).
        normalized = self.ln_2(x)
        # Compute MLP updates; e.g. (2, 5, 768) -> (2, 5, 768).
        update = self.mlp(normalized)
        # Add the MLP residual elementwise; e.g. original 0.7 + update 0.1 = 0.8.
        x = x + update
        return x


# Generate a configuration initializer; e.g. GPTConfig(n_layer=2) overrides just the depth.
@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum context length; e.g. positions 0 through 1023.
    vocab_size: int = 50257  # Number of token IDs; e.g. IDs 0 through 50256.
    n_layer: int = 12  # Transformer depth; e.g. 12 consecutive blocks.
    n_head: int = 12  # Heads per attention layer; e.g. 768 / 12 = 64 features/head.
    n_embd: int = 768  # Embedding width; e.g. each token becomes a vector of 768 numbers.


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Keep model settings available; e.g. self.config.block_size is 1024.
        self.config = config

        # Register named submodules; e.g. transformer.wte accesses the token embedding table.
        self.transformer = nn.ModuleDict(
            dict(
                # Look up token vectors; e.g. 50257 IDs map to rows of a (50257, 768) table.
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Look up position vectors; e.g. 1024 positions map to a (1024, 768) table.
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Register independently initialized blocks; e.g. 12 blocks with separate weights.
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Normalize final token features; e.g. (2, 5, 768) -> (2, 5, 768).
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # Map features to vocabulary scores without an added bias; e.g. (2, 5, 768) -> (2, 5, 50257).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # Scale residual projections; e.g. 12 layers give std = 0.02 / sqrt(24).
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        # Read batch size and token count: (B, T); e.g. 2 sequences of 5 token IDs.
        B, T = idx.size()
        # Reject sequences beyond the context window; e.g. 1025 tokens exceed a limit of 1024.
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}; block size is {self.config.block_size}"
            )

        # Create positions on the input device: (T,); e.g. [0, 1, 2, 3, 4].
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Look up position vectors: (T,) -> (T, C); e.g. (5,) -> (5, 768).
        pos_emb = self.transformer.wpe(pos)
        # Look up token vectors: (B, T) -> (B, T, C); e.g. (2, 5) -> (2, 5, 768).
        tok_emb = self.transformer.wte(idx)
        # Add positions across every batch item: (2, 5, 768) + (5, 768) -> (2, 5, 768).
        x = tok_emb + pos_emb

        # Apply each transformer block in order; e.g. 12 blocks each preserve (2, 5, 768).
        for block in self.transformer.h:
            x = block(x)

        # Normalize each token's features: (B, T, C) -> (B, T, C); e.g. (2, 5, 768).
        x = self.transformer.ln_f(x)
        # Score next-token candidates: (B, T, C) -> (B, T, vocab_size); e.g. (2, 5, 50257).
        logits = self.lm_head(x)
        # Skip loss during generation; e.g. model(idx) returns (logits, None).
        loss = None
        if targets is not None:
            # Flatten batch/time into predictions; e.g. (8, 256, 50257) -> (2048, 50257).
            flat_logits = logits.reshape(-1, logits.size(-1))
            # Match one correct token ID to each prediction; e.g. (8, 256) -> (2048,).
            flat_targets = targets.reshape(-1)
            # Average next-token loss; e.g. uniform predictions give approximately log(50257).
            loss = F.cross_entropy(flat_logits, flat_targets)
        return logits, loss

    # Construct a model from the class; e.g. GPT.from_pretrained("gpt2").
    @classmethod
    def from_pretrained(cls, model_type):
        """Load an original GPT-2 checkpoint from Hugging Face into this model."""
        # Match each checkpoint's architecture; e.g. gpt2 uses 12 layers, 12 heads and 768 features.
        model_configs = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        # Reject unsupported names explicitly; e.g. "gpt3" raises ValueError.
        if model_type not in model_configs:
            raise ValueError(f"Unsupported model type {model_type!r}; choose from {tuple(model_configs)}")

        # Import the checkpoint loader only when needed; e.g. ordinary GPT(config) skips this import.
        from transformers import GPT2LMHeadModel

        # Fix checkpoint vocabulary/context sizes; e.g. 50257 token IDs and 1024 positions.
        config = GPTConfig(vocab_size=50257, block_size=1024, **model_configs[model_type])
        # Build the calling class; e.g. a GPT subclass receives an instance of itself.
        model = cls(config)
        # Share token/output weights as in GPT-2; e.g. both use the same (50257, 768) parameter.
        model.lm_head.weight = model.transformer.wte.weight
        # Access destination tensors by name; e.g. "transformer.h.0.attn.c_attn.weight".
        state_dict = model.state_dict()

        # Fetch the checkpoint or reuse its local cache; e.g. "gpt2" loads the base model.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # Access pretrained tensors by the same names; e.g. "transformer.wte.weight".
        pretrained_state = model_hf.state_dict()

        # Ignore generated attention buffers; e.g. keep our causal mask instead of copying it.
        mask_suffixes = (".attn.bias", ".attn.masked_bias")
        model_keys = {key for key in state_dict if not key.endswith(mask_suffixes)}
        pretrained_keys = {key for key in pretrained_state if not key.endswith(mask_suffixes)}
        # Compare names, not just counts; e.g. a missing ln_f.weight fails before copying.
        if model_keys != pretrained_keys:
            missing = sorted(model_keys - pretrained_keys)
            unexpected = sorted(pretrained_keys - model_keys)
            raise ValueError(f"Checkpoint keys differ: missing={missing}, unexpected={unexpected}")

        # Hugging Face Conv1D stores these matrices in reverse order; e.g. (768, 2304) vs (2304, 768).
        transposed_weights = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        # Copy without recording gradients; e.g. loading weights adds no training graph.
        with torch.no_grad():
            for key in sorted(model_keys):
                # Select the source tensor; e.g. token embeddings have shape (50257, 768).
                weight = pretrained_state[key]
                # Convert Conv1D to Linear layout; e.g. QKV (768, 2304) -> (2304, 768).
                if key.endswith(transposed_weights):
                    weight = weight.t()
                # Require matching dimensions; e.g. a 1024-wide embedding cannot fill a 768-wide one.
                if weight.shape != state_dict[key].shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: {tuple(weight.shape)} != {tuple(state_dict[key].shape)}"
                    )
                # Replace initialized values in place; e.g. copy every layer's learned weights.
                state_dict[key].copy_(weight)

        # Set all modules to evaluation mode; e.g. model.training becomes False.
        model.eval()
        # Return only after all tensors are loaded; e.g. all 12 base-model blocks are populated.
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device, master_process=True):
        # start with all of the candidate parameters(that require grad)
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. ANy parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n,p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameters tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed prameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused=use_fused)
        return optimizer
