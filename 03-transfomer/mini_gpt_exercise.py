"""MiniGPT from scratch — TODO exercise.
Install: pip install torch tiktoken
Each TODO has a corresponding reference in mini_gpt_reference.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# create encoder
enc = tiktoken.get_encoding("cl100k_base")

# B=batch, T=context length, D=d_model, H=num_heads, V=vocab_size.
# Recommended: T=16, D=32, H=4, d_ff=128, num_layers=2, dropout=0.

# TODO 1: Use enc=tiktoken.get_encoding("cl100k_base").
# Encode a small text corpus and build mappings from original token ids
# to contiguous local ids [0, V), and back. Avoid nn.Embedding(1000,32)
# with original token id 15339 — that will be out of range.

def build_data(text, enc):
    # raise NotImplementedError("TODO 1: tokenizer + local vocabulary")
    ids = enc.encode(text)
    # create mapping from original tokenid to local id
    sorted_ids = sorted(set(ids))
    # original id -> local id
    original2local = {
        token_id:local_id
        for local_id, token_id in enumerate(sorted_ids)
    }

    # local id -> original id
    local2original = {
        local_id:token_id
        for local_id, token_id in enumerate(sorted_ids)
    }

    local_ids = [original2local[i] for i in ids]
    data = torch.tensor(local_ids, dtype=torch.long)
    return data, original2local, local2original

def encode_prompt(prompt, enc, original_to_local):
    original_ids = enc.encode(prompt)
    if not original_ids:
        raise ValueError("Prompt must not be empty")
    absent = [i for i in original_ids if i not in original_to_local]    # 将输入转化为本地词表id
    if absent:
        raise ValueError(f"Prompt contains unseen original token ids: {absent}. Try a prompt from the corpus.")
    return torch.tensor([[original_to_local[i] for i in original_ids]], dtype=torch.long)   # 生成训练文本

def decode_local(local_ids, enc, local_to_original):
    original_ids = [local_to_original[int(i)] for i in local_ids]
    # Decode the whole sequence: a single BPE token may be only part of a UTF-8 character.
    return enc.decode(original_ids)

# TODO 2: Given data [a,b,c,d,e,...], sample x=[a,b,c,d] and y=[b,c,d,e].
# Returns x,y both shaped (B,T), torch.long.
def get_batch(data, batch_size, context_size):
    if len(data) <= context_size:
        raise ValueError("Corpus is too short for context_size")
    index = torch.randint(0,len(data) - context_size, (batch_size,), dtype=torch.long)
    list = torch.arange(context_size)
    x = data[index[:,None]+list[None,:]]
    y = data[index[:,None]+list[None,:] +1]
    # [:,None] 将张量升维并广播
    return x, y



# TODO 3: Implement causal multi-head attention.
# x: (B,T,D). q,k,v: (B,H,T,D/H). scores: (B,H,T,T).
# Hint: nn.Linear(D,3*D), .chunk(3,-1), torch.tril, masked_fill.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )
        self.num_heads = num_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        # raise NotImplementedError("TODO 3b: implement causal attention")
        B, T, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(x):
            return x.reshape(B, T, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        B, H, T, Dk = q.shape
        scores = torch.einsum("bhtd,bhsd->bhts",q, k)/math.sqrt(Dk)

        mask = torch.tril(torch.ones(T, T, dtype=bool, device=x.device))
        scores.masked_fill_(~mask, -math.inf)
        scores = self.softmax(scores) @ v
        def combine_heads(x):
            B, H, T, Dk = x.shape
            return x.transpose(1,2).reshape(B,T, -1)
        scores = combine_heads(scores)
        return self.out_proj(scores)



# TODO 4: Linear(D,d_ff) -> GELU -> Linear(d_ff,D).
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # raise NotImplementedError("TODO 4a: define FFN")
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # raise NotImplementedError("TODO 4b: forward FFN")

        return self.W2(self.gelu(self.W1(x)))


# TODO 5: Pre-LN residual: x=x+drop(attn(ln1(x))); x=x+drop(ffn(ln2(x)))
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads=num_heads)
        self.FFN = FeedForward(d_model, d_ff)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        u = x + self.drop_attn(self.attention(self.ln1(x)))
        return u + self.drop_ffn(self.FFN(self.ln2(u)))


# TODO 6: token embedding (V,D), position embedding (T_max,D),
# Decoder blocks x num_layers, final LN, LM Head(D,V).
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context_size=16, d_model=32,
                 num_heads=4, num_layers=2, d_ff=128, dropout=0.0):
        super().__init__()
        self.context_size = context_size
        self.ln_final = nn.LayerNorm(d_model)

        # W_LM \in 32*10000 将高维token向量转化为词表中10000个token的分数
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]) # 多个transformer 堆叠

    def forward(self, input_ids, targets=None):
        # pos = torch.arange(T, device=input_ids.device)
        # x = token_embedding(input_ids) + position_embedding(pos)
        # blocks -> final LN -> LM Head => (B,T,V)
        # optional loss = F.cross_entropy(logits.reshape(B*T,V), targets.reshape(B*T))
        B, T = input_ids.shape
        if T > self.context_size:
            raise ValueError("Sequence too long")
        # 生成position
        position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        #
        logits = self.lm_head(x)
        # 自回归训练：计算预测token与实际目标token的交叉熵
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        else:
            loss = None
        return logits, loss

# TODO 7: autoregressive loop. Crop to last context_size; use logits[:,-1,:]
# then divide by temperature, optionally top-k filter, softmax, multinomial, cat.
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=40, temperature=1.0, top_k=1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()    # 开启推理模式
    for _ in range(max_new_tokens):
        # 只取最后的context_size个token
        logits, _ = model(input_ids[:, -model.context_size:])
        # mini-GPT 逐个预测每个词的下一个词
        new_logits = logits[:,-1,:] / temperature   # 温度越高，softmax后的概率越接近

        # top-k抽样，选择k个可能性最大的token进入候选，然后计算概率
        if top_k is not None:
            top_k = min(top_k, new_logits.shape[-1])
            if top_k < 1:
                raise(ValueError("top_k must be positive"))
            topk_logits, _ = torch.topk(new_logits, top_k, dim=-1)
            threshold = topk_logits[:,[-1]]
            new_logits = new_logits.masked_fill(new_logits < threshold,float("-inf"))
            # topk_logits[:,[-1]] 取最后一位，但是保留原来的维度，也就是 [B, K] -> [B, 1]
            # 将小于阈值的logits全部设为-inf

        probs = torch.softmax(new_logits, dim=-1)

        # 按给定概率分布 probs 做多项式采样，抽 1 个样本索引
        # 为什么不总是选概率最大的？
        # 为了增加生成文本的多样性，否则每次生成的结果都是一样的
        new_id = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, new_id], dim=1)
    return input_ids


def check_model(model,vocab_size):
    model.eval()    # 推理模式
    rd_input = torch.randint(0, vocab_size, (2, 6))
    logits, loss = model(rd_input, rd_input)
    assert loss is not None and torch.isfinite(loss)        # 检查loss是否正常
    model.zero_grad(set_to_none=True)   # 清空模型的所有可训练参数的梯度
    loss.backward()
    assert model.blocks[0].attention.qkv.weight.grad is not None # 检查attention有没有梯度
    assert model.token_embedding.weight.grad is not None    # 检查token_embedding有无梯度（embedding本身也是可训练的参数）
    assert model.position_embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    with torch.no_grad():
        one = torch.randint(0, vocab_size - 1, (1,6))
        two = one.clone()
        two[:,4:] = two[:,4:] + 1
        output_1, _ = model(one)
        output_2, _ = model(two)
        early_diff = (output_1[:,:4] - output_2[:,:4]).abs().max()
        print("Early position max diff:",early_diff)
        assert torch.allclose(
            output_1[:, :4, :],
            output_2[:, :4, :],
            atol=1e-6
        ), "Causal mask test failed!"

        print("All tests passed!")



# TODO 8: construct corpus, use tiktoken + local vocab, model,
# AdamW optimizer, zero_grad/backward/step. Re-map generated ids to
# original ids before enc.decode(...).
def main():
    enc = tiktoken.get_encoding("cl100k_base")
    corpus = ("hello world! this is a tiny gpt model.\n hello, i can see u.\n") * 250
    data, original_to_local, local_to_original = build_data(corpus, enc)
    vocab_size = len(local_to_original)
    print(f"number of corpus tokens={len(data)}, local vocab_size={vocab_size}, original vocab={enc.n_vocab}")
    model = MiniGPT(vocab_size=vocab_size)
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    check_model(model, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(1,201):
        model.train()
        xb, yb = get_batch(data, batch_size=16, context_size=model.context_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:03d} | train loss {loss.item():.4f}")

    prompt_ids = encode_prompt("hello", enc, original_to_local)
    generated = generate(model, prompt_ids, max_new_tokens=40, temperature=1.0, top_k=1)
    print("generated:", repr(decode_local(generated[0].tolist(), enc, local_to_original)))


if __name__ == "__main__":
    main()
