"""Train MiniGPT on explicit question/answer records with answer-only loss.

Run from the repository root:
    python "03-transformer/11-train-minigpt.py" --steps 1200
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from minigpt_cli_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_QA,
    END_TOKEN,
    UNK_TOKEN,
    choose_device,
    filter_model_config,
    load_minigpt_class,
    normalize_question,
)


# Keep these examples in training so the first command-line checks are deterministic
# demonstrations of memorization, not held-out generalization tests.
FORCE_TRAIN_QUESTIONS = {"你好", "什么是人工智能", "什么是top-k"}


def load_qa(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是有效 JSONL：{exc}") from exc
            if "question" not in item or "answer" not in item:
                raise ValueError(f"{path}:{line_no} 必须包含 question 和 answer 字段")
            question = normalize_question(str(item["question"]))
            answer = str(item["answer"]).strip()
            if question and answer:
                records.append({"question": question, "answer": answer})
    if len(records) < 5:
        raise ValueError("问答数据少于 5 条，请检查 JSONL 文件。")
    return records


def split_records(records: list[dict[str, str]], val_ratio: float, seed: int):
    """Split whole QA records, keeping every validation character represented in train."""
    if not 0 <= val_ratio < 0.5:
        raise ValueError("--val-ratio 必须在 [0, 0.5) 范围内")
    if val_ratio == 0:
        return list(records), []

    force_train = []
    candidates = []
    for record in records:
        if record["question"] in FORCE_TRAIN_QUESTIONS:
            force_train.append(record)
        else:
            candidates.append(record)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    target_val = max(1, round(len(records) * val_ratio))

    # Document frequency: count in how many QA records each character appears.
    char_record_freq = Counter()
    record_chars = []
    for record in records:
        chars = set(record["question"] + record["answer"] + "用户：\n助手：")
        record_chars.append(chars)
        char_record_freq.update(chars)

    val_records = []
    selected_ids = set()
    for record in candidates:
        if len(val_records) >= target_val:
            break
        chars = set(record["question"] + record["answer"] + "用户：\n助手：")
        # Do not hold out a record if doing so would remove a character from training vocab.
        if all(char_record_freq[ch] > 1 for ch in chars):
            val_records.append(record)
            selected_ids.add(id(record))
            for ch in chars:
                char_record_freq[ch] -= 1

    train_records = [r for r in records if id(r) not in selected_ids]
    return train_records, val_records


def build_vocab(train_records: list[dict[str, str]]) -> list[str]:
    chars = set("用户：\n助手：")
    for record in train_records:
        chars.update(record["question"])
        chars.update(record["answer"])
    # Ordinary characters are single-character tokens; these two are atomic special tokens.
    return sorted(chars) + [UNK_TOKEN, END_TOKEN]


def encode_record(record, token_to_id):
    prompt = f"用户：{record['question']}\n助手："
    prompt_ids = [token_to_id.get(ch, token_to_id[UNK_TOKEN]) for ch in prompt]
    answer_ids = [
        token_to_id.get(ch, token_to_id[UNK_TOKEN]) for ch in record["answer"]
    ]
    ids = prompt_ids + answer_ids + [token_to_id[END_TOKEN]]
    return {"ids": ids, "prompt_len": len(prompt_ids), "question": record["question"]}


def make_window(encoded, context_size, pad_id, rng=None, random_start=True):
    """Return x and next-token targets; prompt targets are masked with -100."""
    ids = encoded["ids"]
    prompt_len = encoded["prompt_len"]
    if prompt_len >= context_size:
        raise ValueError(
            f"问题提示长度 {prompt_len} 已达到 context_size={context_size}；"
            "请增大 --context-size。"
        )

    input_length = len(ids) - 1  # inputs predict the next token; EOS is a target, not an input.
    max_start = max(0, input_length - context_size)
    if max_start == 0 or not random_start or rng is None:
        start = 0
    elif rng.random() < 0.5:
        start = 0  # Often train from the question prefix so the answer mapping is learned.
    else:
        start = rng.randint(0, max_start)

    # x contains only positions that have a next-token target; the final EOS
    # is predicted by the preceding token, not used as an input in this sample.
    x = ids[start:min(start + context_size, input_length)]
    raw_targets = ids[start + 1:min(start + context_size + 1, len(ids))]
    targets = []
    for offset, token_id in enumerate(raw_targets):
        absolute_target_position = start + 1 + offset
        targets.append(token_id if absolute_target_position >= prompt_len else -100)

    if len(x) < context_size:
        x += [pad_id] * (context_size - len(x))
    if len(targets) < context_size:
        targets += [-100] * (context_size - len(targets))

    return torch.tensor(x, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def sample_batch(encoded_records, batch_size, context_size, pad_id, device, rng):
    xs, ys = [], []
    for _ in range(batch_size):
        record = rng.choice(encoded_records)
        x, y = make_window(record, context_size, pad_id, rng=rng, random_start=True)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate_loss(model, encoded_records, batch_size, context_size, pad_id, device):
    """Deterministic evaluation: one prefix-aligned window per QA, answer tokens only."""
    if not encoded_records:
        return None
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(encoded_records), batch_size):
        chunk = encoded_records[start:start + batch_size]
        pairs = [
            make_window(r, context_size, pad_id, random_start=False)
            for r in chunk
        ]
        x = torch.stack([p[0] for p in pairs]).to(device)
        y = torch.stack([p[1] for p in pairs]).to(device)
        logits, _ = model(x)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += int((y != -100).sum().item())
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="Train a QA-oriented character-level MiniGPT.")
    parser.add_argument("--data", type=Path, default=DEFAULT_QA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1:
        raise ValueError("--steps、--batch-size 和 --eval-interval 必须为正整数")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model 必须能被 --num-heads 整除")
    if not args.data.exists():
        raise FileNotFoundError(f"找不到问答数据：{args.data}")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device()

    records = load_qa(args.data)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    tokens = build_vocab(train_records)
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    encoded_train = [encode_record(r, token_to_id) for r in train_records]
    encoded_val = [encode_record(r, token_to_id) for r in val_records]
    pad_id = token_to_id.get(UNK_TOKEN, 0)

    # Fail early rather than silently training on examples that have no answer targets.
    for encoded in encoded_train + encoded_val:
        if encoded["prompt_len"] >= args.context_size:
            raise ValueError(
                f"问题 {encoded['question']!r} 的提示长度超过上下文窗口。"
                "请增加 --context-size。"
            )

    requested_config = {
        "context_size": args.context_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }
    MiniGPT = load_minigpt_class()
    config = filter_model_config(MiniGPT, requested_config)
    if "context_size" not in config:
        raise TypeError(
            "MiniGPT 构造函数没有 context_size 参数；请检查 08-mini-gpt.py 的模型接口。"
        )
    model = MiniGPT(vocab_size=len(tokens), **config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"QA records: {len(records)}")
    print(f"Train/validation records: {len(train_records)}/{len(val_records)}")
    print(f"Vocabulary tokens: {len(tokens)} (includes atomic {END_TOKEN} and {UNK_TOKEN})")
    print(f"Model parameters: {num_params:,}")
    print(f"Model config actually used: {config}")
    print("Loss: answer characters + end token only; prompt-token targets are ignored.")
    print("Training...")

    best_metric = float("inf")
    best_step = 0
    best_train_loss = None
    best_val_loss = None

    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(
            encoded_train, args.batch_size, args.context_size, pad_id, device, rng
        )
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = evaluate_loss(
                model, encoded_train, args.batch_size, args.context_size,
                pad_id, device
            )
            val_loss = evaluate_loss(
                model, encoded_val, args.batch_size, args.context_size,
                pad_id, device
            )
            metric_name = "val" if val_loss is not None else "train"
            metric = val_loss if val_loss is not None else train_loss
            print(
                f"step {step:4d}/{args.steps} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}" if val_loss is not None else
                f"step {step:4d}/{args.steps} | train loss {train_loss:.4f} | val: disabled"
            )

            if metric < best_metric:
                best_metric = metric
                best_step = step
                best_train_loss = train_loss
                best_val_loss = val_loss
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "format_version": 2,
                    "model_state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "tokens": tokens,
                    "config": config,
                    "best_step": best_step,
                    "total_steps": args.steps,
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "selection_metric": metric_name,
                    "num_records": len(records),
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "seed": args.seed,
                }
                torch.save(checkpoint, args.checkpoint)

    print(f"\nBest checkpoint saved to: {args.checkpoint}")
    print(f"Best step: {best_step} | selection metric: {best_metric:.4f}")
    if best_val_loss is not None:
        print(f"Train loss at best checkpoint: {best_train_loss:.4f}")
        print(f"Validation loss at best checkpoint: {best_val_loss:.4f}")
    print("\nNext command:")
    print('python "03-transformer/12-chat-minigpt.py" --top-k 1 --no-history')


if __name__ == "__main__":
    main()
