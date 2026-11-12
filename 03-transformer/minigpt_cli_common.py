"""Shared utilities for the QA-trained character-level MiniGPT."""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_QA = HERE / "data" / "minigpt_qa.jsonl"
DEFAULT_CHECKPOINT = HERE / "minigpt_qa_checkpoint.pt"
MODEL_SOURCE = HERE / "08-mini-gpt.py"

END_TOKEN = "<|eot|>"   # One dedicated token: end of answer.
UNK_TOKEN = "<|unk|>"   # One dedicated token: unseen input character.


def normalize_question(text: str) -> str:
    """Normalize spaces between Chinese characters and Latin/digit runs."""
    text = text.strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def load_minigpt_class():
    """Import MiniGPT from the existing 08-mini-gpt.py file."""
    if not MODEL_SOURCE.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_SOURCE}\n"
            "请确认本项目放在仓库的 03-transformer 目录，并保留原来的 08-mini-gpt.py。"
        )
    spec = importlib.util.spec_from_file_location("ai_roadmap_mini_gpt", MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型文件：{MODEL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MiniGPT"):
        raise AttributeError("08-mini-gpt.py 中没有找到名为 MiniGPT 的类。")
    return module.MiniGPT


def filter_model_config(MiniGPT, requested: dict) -> dict:
    """Pass only constructor options supported by this repository's MiniGPT."""
    params = inspect.signature(MiniGPT.__init__).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return dict(requested)
    return {key: value for key, value in requested.items() if key in params}


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load the QA checkpoint and recreate the exact model/token vocabulary."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}\n"
            "请先运行 11-train-minigpt.py 训练新版 QA 模型。"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("format_version") != 2 or "tokens" not in checkpoint:
        raise ValueError(
            "这是旧版字符级 checkpoint，不能直接用于新版 QA 训练脚本。"
            "请运行 11-train-minigpt.py 生成 minigpt_qa_checkpoint.pt。"
        )

    tokens = checkpoint["tokens"]
    config = checkpoint["config"]
    MiniGPT = load_minigpt_class()
    model = MiniGPT(vocab_size=len(tokens), **config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return model, tokens, token_to_id, config, checkpoint


def encode_text(text: str, token_to_id: dict[str, int]) -> list[int]:
    """Encode ordinary text one character at a time; unknown characters use UNK."""
    unk_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(char, unk_id) for char in text]


def decode_ids(token_ids, tokens: list[str]) -> str:
    """Decode character IDs; internal control tokens are not shown to the user."""
    pieces = []
    for token_id in token_ids:
        token = tokens[int(token_id)]
        if token in {END_TOKEN, UNK_TOKEN}:
            if token == UNK_TOKEN:
                pieces.append("�")
            continue
        pieces.append(token)
    return "".join(pieces)
