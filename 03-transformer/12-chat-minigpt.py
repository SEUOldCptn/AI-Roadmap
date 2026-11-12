"""Command-line QA interaction with the answer-masked MiniGPT checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minigpt_cli_common import (
    DEFAULT_CHECKPOINT,
    END_TOKEN,
    UNK_TOKEN,
    choose_device,
    decode_ids,
    encode_text,
    load_checkpoint,
    normalize_question,
)


@torch.inference_mode()
def generate_answer(
    model, prompt_ids, tokens, token_to_id, context_size,
    max_new_tokens, temperature, top_k, device
):
    """Autoregressively generate until the dedicated end-of-answer token appears."""
    end_id = token_to_id[END_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        context = input_ids[:, -context_size:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature

        # UNK is for encoding unseen input characters, not a desirable output.
        next_logits[:, unk_id] = float("-inf")

        if top_k is not None:
            k = min(int(top_k), next_logits.size(-1))
            values, _ = torch.topk(next_logits, k=k, dim=-1)
            next_logits = next_logits.masked_fill(
                next_logits < values[:, [-1]], float("-inf")
            )

        if top_k == 1:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)

        token_id = int(next_id.item())
        if token_id == end_id:
            break

        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return decode_ids(generated, tokens)


def main():
    parser = argparse.ArgumentParser(description="Chat with the QA-trained MiniGPT.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=1, help="1=贪心解码；0=不限制候选数")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--history", action="store_true", help="将之前轮次加入上下文")
    parser.add_argument("--no-history", action="store_true", help="兼容旧命令；默认本来就不带历史")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.temperature <= 0:
        raise ValueError("--temperature 必须大于 0")
    if args.top_k < 0:
        raise ValueError("--top-k 不能小于 0")
    if args.history and args.no_history:
        raise ValueError("--history 和 --no-history 不能同时使用")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device()
    model, tokens, token_to_id, config, checkpoint = load_checkpoint(
        args.checkpoint, device
    )
    context_size = int(config["context_size"])
    top_k = None if args.top_k == 0 else args.top_k
    use_history = args.history and not args.no_history

    print("\nMiniGPT QA 命令行交互已启动")
    print(
        f"Device: {device} | vocab: {len(tokens)} | "
        f"best step: {checkpoint.get('best_step', '?')}"
    )
    print("输入问题；/reset 清空历史；/exit 或 /quit 退出；/help 查看命令。")
    print("建议先用 --top-k 1 测试训练样本的复现能力。\n")

    history = ""
    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出 MiniGPT。")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("已退出 MiniGPT。")
            break
        if user_text == "/reset":
            history = ""
            print("（对话历史已清空）")
            continue
        if user_text == "/help":
            print("命令：/reset 清空历史；/exit 或 /quit 退出。默认每轮独立；--history 开启历史。")
            continue

        question = normalize_question(user_text)
        prompt_text = f"{history}用户：{question}\n助手："
        prompt_ids = encode_text(prompt_text, token_to_id)
        if len(prompt_ids) >= context_size:
            prompt_ids = prompt_ids[-(context_size - 1):]

        answer = generate_answer(
            model=model,
            prompt_ids=prompt_ids,
            tokens=tokens,
            token_to_id=token_to_id,
            context_size=context_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            device=device,
        )

        # Safety net for a model that emits a new user-turn marker before EOS.
        for marker in ("\n用户：", "用户："):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        answer = answer.strip()
        print(f"助手：{answer or '（模型立即生成了结束标记；请继续训练或检查 checkpoint。）'}")

        if use_history:
            history = (history + f"用户：{question}\n助手：{answer}\n")[-max(1, context_size // 2):]


if __name__ == "__main__":
    main()
