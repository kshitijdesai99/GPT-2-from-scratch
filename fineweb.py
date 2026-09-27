import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100 million tokens per full shard; the last can be smaller.
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

# Initialize lazily in each worker; importing this module does not fetch tokenizer data.
enc = None


def tokenize(doc):
    global enc
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    # Prefix each document with its boundary marker; e.g. GPT-2's end-of-text ID is 50256.
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # Ensure IDs fit in two bytes; e.g. 50256 is below the uint16 limit of 65536.
    if not ((0 <= tokens_np).all() and (tokens_np < 2**16).all()):
        raise ValueError("Token IDs must fit in uint16")
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    # Write raw uint16 bytes; read back with np.fromfile(filename, dtype=np.uint16).
    # Only completed shards get their final name; e.g. interrupted writes remain .tmp files.
    temporary = filename + ".tmp"
    with open(temporary, "wb") as f:
        f.write(tokens_np.tobytes())
    os.replace(temporary, filename)


def main():
    # Create output storage and load the dataset only when this script is executed directly.
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # Use half the available CPUs; fall back to one worker if the count is unknown.
    nprocs = max(1, (os.cpu_count() or 1) // 2)
    shard_index = 0
    # Reserve one shard buffer; e.g. 100 million uint16 tokens occupy 200 MB.
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    try:
        with mp.Pool(nprocs) as pool:
            for tokens in pool.imap(tokenize, fw, chunksize=16):
                offset = 0
                # A document may span several shards; keep copying until all its tokens fit.
                while offset < len(tokens):
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

                    # Copy only what fits; e.g. 3 free slots and 8 remaining tokens means copy 3.
                    count = min(shard_size - token_count, len(tokens) - offset)
                    all_tokens_np[token_count:token_count + count] = tokens[offset:offset + count]
                    token_count += count
                    offset += count
                    progress_bar.update(count)

                    if token_count == shard_size:
                        # Reserve shard 0 for validation; subsequent shards are training data.
                        split = "val" if shard_index == 0 else "train"
                        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                        write_datafile(filename, all_tokens_np)
                        shard_index += 1
                        token_count = 0
                        progress_bar.close()
                        progress_bar = None

            # Write only valid tokens in the final partial shard; e.g. 60M of 100M slots.
            if token_count:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                write_datafile(filename, all_tokens_np[:token_count])
    finally:
        if progress_bar is not None:
            progress_bar.close()


# Spawned workers import this file without starting another download or process pool.
if __name__ == "__main__":
    main()
