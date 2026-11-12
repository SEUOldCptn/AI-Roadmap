# MiniGPT 问答训练升级版

这个版本保留现有 `08-mini-gpt.py` 的 MiniGPT
架构，只升级数据与训练/生成流程。

## 文件

-   `11-train-minigpt.py`：从 JSONL 问答样本训练模型，答案部分才计算
    loss，保存验证损失最低的 checkpoint。
-   `12-chat-minigpt.py`：命令行问答，遇到专用 `<|eot|>` token
    就停止生成。
-   `minigpt_cli_common.py`：模型加载、词表、文本归一化和 checkpoint
    工具。
-   `data/minigpt_qa.jsonl`：从原中文教学语料整理出的独立问答记录。

请保留原来的 `08-mini-gpt.py`。本版本会生成新的
`minigpt_qa_checkpoint.pt`，不会覆盖旧的 `minigpt_char_checkpoint.pt`。

## 1. 放置文件

把 ZIP 解压到 AI-Roadmap 仓库根目录，让其中的 `03-transfomer`
文件夹与仓库同名文件夹合并。允许覆盖这三个文件：

-   `03-transfomer/11-train-minigpt.py`
-   `03-transfomer/12-chat-minigpt.py`
-   `03-transfomer/minigpt_cli_common.py`

并新增：

-   `03-transfomer/data/minigpt_qa.jsonl`

不要覆盖 `08-mini-gpt.py`。

## 2. 先做 100 步冒烟训练

在 PowerShell / PyCharm Terminal 的仓库根目录运行：

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 100 --eval-interval 25
```

确认模型能够完成前向传播、答案掩码损失、验证集评估和 checkpoint 保存。

## 3. 正式训练

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 1200 --eval-interval 100
```

默认设置：`context_size=128`、`d_model=32`、`num_layers=1`、`dropout=0.1`、`lr=0.001`。脚本会检测
`MiniGPT.__init__` 支持哪些参数，只有模型构造函数接受的参数才会传入。

默认会按完整问答记录切分训练/验证集，并确保验证集中的字符也能在训练集词表中找到。`你好`、`什么是人工智能`、`什么是top-k`
这几个常用检查问题会留在训练集中，用于演示模型是否记住训练样本。它们不是泛化测试题。

如果想让所有记录都参与训练（用于专门观察记忆/过拟合），可执行：

``` powershell
python "03-transfomer/11-train-minigpt.py" --steps 1200 --val-ratio 0
```

当 `--val-ratio 0` 时没有独立验证集，checkpoint
将依据训练损失选择；不要把这个结果当成泛化能力的证据。

## 4. 命令行交互

``` powershell
python "03-transfomer/12-chat-minigpt.py" --top-k 1 --no-history
```

建议先用 `--top-k 1`
做贪心解码，便于检查模型是否复现训练样本。`temperature` 在 `top-k=1`
时不会改变 argmax 选择。

也可以尝试：

``` powershell
python "03-transfomer/12-chat-minigpt.py" --max-new-tokens 120 --temperature 0.7 --top-k 8
```

-   `--max-new-tokens`：最多生成多少字符 token。
-   `--temperature`：采样温度。
-   `--top-k 1`：贪心解码。
-   `--top-k 0`：不限制候选数量。
-   默认每轮不携带历史；`--history` 开启历史。
-   `/reset` 清空历史，`/exit` 退出。

## 5. 关键实现逻辑

### 独立 QA 样本

每条记录格式为 JSONL：

``` json
{"question":"什么是人工智能","answer":"人工智能研究如何让计算机完成感知、学习、推理和决策等任务。"}
```

模型训练序列是：

`用户：问题\n助手：答案<|eot|>`

`<|eot|>` 是一个原子 token，不是 5 个或更多个普通字符。生成时遇到这个
token 就停止。

### 答案部分 loss masking

输入 `x` 与目标 `y` 仍然错开一个
token。若目标位置属于问题提示部分，就将该目标设为 `-100`；PyTorch 的
`cross_entropy(ignore_index=-100)` 会忽略这些位置。答案字符与 `<|eot|>`
仍然参与 loss。

因此，问题字符作为上下文参与前向传播，但不会直接作为预测目标计算 loss。

### 为什么仍可能回答不好

这仍然是从随机参数开始训练的字符级小模型，数据只有几十条问答，没有预训练或真正的大规模指令微调。该版本更适合学习"问答格式、loss
masking、EOS 停止和 checkpoint 选择"，不应期待通用助手能力。

## 6. Git 操作

先检查工作区，避免覆盖未提交修改：

``` powershell
git status
git switch -c minigpt-qa-upgrade
```

将文件放好并测试后：

``` powershell
git add "03-transfomer/11-train-minigpt.py"
git add "03-transfomer/12-chat-minigpt.py"
git add "03-transfomer/minigpt_cli_common.py"
git add "03-transfomer/data/minigpt_qa.jsonl"
git add "03-transfomer/README_minigpt_qa_upgrade.md"
git commit -m "feat: add answer-masked MiniGPT QA training"
git push -u origin minigpt-qa-upgrade
```

然后在 GitHub 上创建 Pull Request。不要提交生成的 `.pt` checkpoint。
