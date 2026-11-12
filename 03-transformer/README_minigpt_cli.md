# MiniGPT 语料训练与命令行交互

这个小项目复用仓库已有的
`03-transfomer/08-mini-gpt.py`，训练一个真正执行前向传播、计算交叉熵并通过反向传播更新参数的字符级
MiniGPT。

## 文件结构

``` text
03-transfomer/
├── 08-mini-gpt.py                 # 已有的 MiniGPT 模型定义
├── minigpt_cli_common.py          # 模型加载、字符编码和 checkpoint 工具
├── 11-train-minigpt.py            # 训练脚本
├── 12-chat-minigpt.py             # 命令行交互脚本
├── minigpt_char_checkpoint.pt     # 训练后自动生成，不包含在本项目压缩包里
└── data/
    └── minigpt_corpus.txt         # 原创中文教学语料
```

## 1. 环境准备

在 Windows PowerShell 或 PyCharm Terminal 中，从仓库根目录执行：

``` powershell
python --version
python -m pip install torch
```

本项目不要求独立显卡。脚本会在 CUDA 可用时使用 CUDA，否则使用 CPU。

## 2. 先做一次快速训练

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 100
```

快速训练用于确认语料、模型导入、前向传播、损失计算和 checkpoint
保存流程都正常。

## 3. 进行更完整的训练

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 1000
```

训练时会周期性显示 train loss 和 validation
loss。训练结束后，模型权重和字符表会保存到：

`03-transfomer/minigpt_char_checkpoint.pt`

可调整的参数示例：

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 1500 --batch-size 16 --context-size 64 --lr 0.003
```

注意：`--context-size` 必须与训练时的配置一起保存在 checkpoint
中，交互脚本会自动读取该配置。

## 4. 启动命令行交互

``` powershell
python "03-transfomer/12-chat-minigpt.py"
```

可调整生成长度、温度和 top-k：

``` powershell
python "03-transfomer/12-chat-minigpt.py" --max-new-tokens 100 --temperature 0.8 --top-k 8
```

-   `--max-new-tokens`：每轮最多生成多少个字符 token。
-   `--temperature`：控制采样随机性，必须大于 0。
-   `--top-k`：只从概率最高的 k 个候选中采样；设置为 0 表示不限制。
-   `--no-history`：每轮只使用当前输入，不携带之前的对话。

交互中可输入： - `/reset`：清空对话历史。 - `/help`：查看命令。 -
`/exit` 或 `/quit`：退出程序。

## 5. 模型和 tokenizer 的设计

-   模型：复用 `08-mini-gpt.py` 中的 Transformer
    Decoder、因果注意力、位置 Embedding、LayerNorm、FFN
    和语言模型输出层。
-   Tokenizer：字符级 tokenizer，每个字符映射为一个整数 ID。
-   训练目标：输入文本与向后移动一个字符的目标序列，学习预测下一个字符。
-   优化器：AdamW。
-   验证：每隔固定步数报告 train loss 和 validation loss。
-   Checkpoint：保存
    `model_state`、字符表和模型配置，确保推理时使用与训练一致的映射。

用户输入中不在语料词表里的字符会映射为 `？`。字符级 tokenizer
仅用于教学，真实 LLM 通常会采用更成熟的子词 tokenizer。

## 6. 重要预期

这是从随机参数开始训练的小模型，不是预训练模型，也没有进行指令微调。它可能模仿语料中的问答格式，但不能保证理解问题或提供可靠事实。模型生成结果不佳并不代表
Transformer
原理错误；可以先检查损失曲线，再尝试增加训练步数、扩充语料或调整模型规模。

本项目使用 CPU 也能运行，但训练速度会受电脑性能影响。建议先运行 100
步验证完整流程，再运行 1000 步观察效果。

## 7. Git 提交流程

``` powershell
git status
git switch main
git pull --ff-only origin main
git switch -c minigpt-corpus-cli
```

把本项目文件放入对应目录后：

``` powershell
git add "03-transfomer/minigpt_cli_common.py"
git add "03-transfomer/11-train-minigpt.py"
git add "03-transfomer/12-chat-minigpt.py"
git add "03-transfomer/data/minigpt_corpus.txt"
git add "03-transfomer/README_minigpt_cli.md"
git commit -m "feat: train and chat with character-level MiniGPT"
git push -u origin minigpt-corpus-cli
```

然后在 GitHub 上创建 Pull Request。训练生成的 `.pt` checkpoint
不需要提交到 Git。
