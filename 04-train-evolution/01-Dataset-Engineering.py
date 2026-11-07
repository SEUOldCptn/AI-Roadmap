
"""
Phase A - Week 01
MiniGPT Dataset Engineering

Run:
python 04-training-evaluation/01-dataset-engineering.py

Requirements:
pip install torch tiktoken
"""

import torch
import tiktoken


CONTEXT_SIZE = 16
BATCH_SIZE = 8
SEED = 42


# 用不同内容的文档替代原来重复250次的语料
DOCUMENTS = [
    "An astronomer measures the brightness of a distant star.",
    "A satellite sends telemetry to a ground station.",
    "A mobile robot estimates its position using camera images.",
    "The weather station records temperature and wind speed.",
    "A neural network learns patterns from training examples.",
    "The engineer checks model outputs before updating weights.",
    "A language model predicts the next token from context.",
    "The training dataset contains documents with different topics.",
    "The validation set evaluates performance on unseen documents.",
    "The test set is reserved for final model evaluation.",
    "A data pipeline removes duplicate records before tokenization.",
    "The optimizer updates parameters using gradients.",
    "The attention layer combines information from previous tokens.",
    "A cache stores keys and values to avoid repeated computation.",
    "The server groups requests to improve hardware utilization.",
    "The profiler identifies operations that consume execution time.",
    "A GPU executes many threads in parallel.",
    "The memory system transfers tensors between storage levels.",
    "The researcher compares latency and throughput.",
    "A reproducible experiment records its configuration and seed."
]


# TODO 1:
# Split document indices into train / val / test.
#
# Requirements:
# 1. Train: 80%, Val: 10%, Test: remaining 10%.
# 2. Use torch.randperm().
# 3. Use a fixed random seed.
# 4. Return a dictionary:
#    {"train": [...], "val": [...], "test": [...]}
# 5. No document index can appear in two splits.

def split_documents(
    documents,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    generator = torch.Generator().manual_seed(seed)
    num = len(documents)
    train_num = int(num * train_ratio)
    val_num = int(num * val_ratio)
    perm = torch.randperm(num,generator=generator).tolist()

    return {
        "train":perm[:train_num],
        "val":perm[train_num:train_num+val_num],
        "test":perm[train_num+val_num:]
            }




# TODO 2:
# Tokenize each document separately.
# Append enc.eot_token after each document.
#
# Return:
# {
#   "train": [[...], [...]],
#   "val": [[...]],
#   "test": [[...]]
# }

def tokenize_splits(split_indices, documents, enc):
    tokenized_documents = {}
    for split_name, indices in split_indices.items():
        tokenized_documents[split_name] = []
        # 遍历当前key中的文档索引
        for index in indices:
            text = documents[index]
            token_ids = enc.encode(text)
            # 添加文档结束标记
            token_ids.append(enc.eot_token)
            # 存到对应key中的list
            tokenized_documents[split_name].append(token_ids)
    return tokenized_documents


# TODO 3:
# Build a compact mapping:
# original_token_id -> local_token_id
#
# Keep the existing MiniGPT compact vocabulary design.
# This mapping is only for compact indexing, not model training.

def build_local_vocab(tokenized_splits):
    local_ids = {}
    local_indice = 0
    for name, split_indices in tokenized_splits.items():
        for list in split_indices:
            for list_index in list:
                if list_index not in local_ids:
                    local_ids[list_index] = local_indice
                    local_indice += 1


    return local_ids


# TODO 4:
# Flatten documents within each split separately.
# Do not concatenate train, val and test together.

def pack_split(tokenized_documents, token_to_local):
    local_ids = []
    for doc in tokenized_documents:
        local_ids.extend([token_to_local[id] for id in doc])
        # append()：把整个列表作为一个元素添加进去。
        # extend()：把列表内的所有元素依次添加进去。
    return torch.tensor(local_ids, dtype=torch.long)



# TODO 5:
# Implement next-token sampling.
#
# Input:
# data: (N,)
#
# Output:
# x: (B,T)
# y: (B,T)
#
# Remember y is shifted by one position.

def get_batch(data, batch_size, context_size):
    x_sample = []
    y_sample = []
    if len(data) <= context_size:
        raise ValueError("Input data too short!")
    start = torch.randint(0, len(data) - context_size + 1, (batch_size,), dtype=torch.long)
    for idx in start:
        x_sample.append(data[idx:idx + context_size])
        y_sample.append(data[idx+1:idx + context_size+1])
    x_sample = torch.stack(x_sample)
    y_sample = torch.stack(y_sample)
    return x_sample, y_sample



def main():

    torch.manual_seed(SEED)
    enc = tiktoken.get_encoding("cl100k_base")

    split_indices = split_documents(DOCUMENTS)

    # TODO 6: Check that the three sets are disjoint.

    tokenized = tokenize_splits(
        split_indices, DOCUMENTS, enc
    )

    token_to_local = build_local_vocab(tokenized)

    datasets = {
        name: pack_split(tokenized[name], token_to_local)
        for name in ["train", "val", "test"]
    }

    print("Vocabulary size:", len(token_to_local))

    for name, data in datasets.items():
        print(
            f"{name}: "
            f"documents={len(split_indices[name])}, "
            f"tokens={len(data)}"
        )

    # TODO 7: Test get_batch using train data only.

    x, y = get_batch(
        datasets["train"],
        BATCH_SIZE,
        CONTEXT_SIZE
    )

    assert x.shape == (BATCH_SIZE, CONTEXT_SIZE)
    assert y.shape == (BATCH_SIZE, CONTEXT_SIZE)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # TODO 8: Verify the one-token shift.

    assert torch.equal(x[:, 1:], y[:, :-1])

    print("All dataset tests passed!")


if __name__ == "__main__":
    main()
