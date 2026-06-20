import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """  
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    维度约定：
    Q, K, V: (batch_size, n_heads, seq_len, d_k)
    返回:    (batch_size, n_heads, seq_len, d_k)    
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # 1. 计算 raw scores: Q @ K^T
        #    shape: (B, H, L, d_k) @ (B, H, d_k, L) → (B, H, L, L)
        attn_weights = Q @ K.transpose(-2, -1)
        # 2. scale: / sqrt(d_k)
        attn_weights = attn_weights / math.sqrt(self.d_k)
        # 3. mask (如果给的话): scores.masked_fill(mask == 0, -1e9)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        # 4. softmax over the last dim → attn_weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        # 5. attn_weights @ V → output  
        output = attn_weights @ V

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    多头 = 多组 QKV 投影 + 并行 attention + 拼接投影回 d_model

    Q = X @ W_q, K = X @ W_k, V = X @ W_v  ← 都来自同一个输入 X
    投影后 reshape 成 (B, L, n_heads, d_k) 再 transpose
    """ 
    def __init__(self, d_model= 512, n_heads=8, d_k=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        # W_q, W_k, W_v: 把 X 从 d_model 投影到 n_heads * d_k
        self.W_q = nn.Linear(d_model, n_heads * d_k)
        self.W_k = nn.Linear(d_model, n_heads * d_k)
        self.W_v = nn.Linear(d_model, n_heads * d_k)
        # W_o: 拼接后投影回 d_model
        self.W_o = nn.Linear(n_heads * d_k, d_model)
        self.attention = ScaledDotProductAttention(d_k)        

    def forward(self, X, mask=None):
        B, L, D = X.shape
        # 1. 线性投影: Q = X @ W_q, 同理 K, V
        #    每个 shape: (B, L, n_heads * d_k)       
        Q =  self.W_q(X)
        K =  self.W_k(X)
        V =  self.W_v(X)
        # 2. reshape + transpose:
        #    (B, L, n_heads * d_k) → (B, L, n_heads, d_k) → (B, n_heads, L, d_k)        
        Q = Q.reshape(B, L, self.n_heads, self.d_k)
        Q = Q.transpose(1, 2)
        K = K.reshape(B, L, self.n_heads, self.d_k)
        K = K.transpose(1, 2)
        V = V.reshape(B, L, self.n_heads, self.d_k)
        V = V.transpose(1, 2)

        # 3. ScaledDotProductAttention
        output, attn_weights = self.attention(Q, K, V, mask)
        # 4. transpose 回去 + reshape:
        #    (B, n_heads, L, d_k) → (B, L, n_heads, d_k) → (B, L, n_heads * d_k)
        output = output.transpose(1, 2)
        output = output.reshape(B, L, -1)
        # 5. 投影: W_o → (B, L, d_model)
        output = self.W_o(output)
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 创建 pe = zeros(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 2. position = arange(0, max_len).unsqueeze(1)  → (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 3. div_term = exp(arange(0, d_model, 2) * (-log(10000) / d_model))  → (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 4. pe[:, 0::2] = sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        #    pe[:, 1::2] = cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 5. pe = pe.unsqueeze(0)  → (1, max_len, d_model) 广播到 batch
        pe = pe.unsqueeze(0)
        # 6. register_buffer('pe', pe)  ← 不是 Parameter，不参与梯度  
        self.register_buffer('pe', pe)

    def forward(self, x):
    # x: (B, seq_len, d_model)
    # pe[:, :x.size(1), :] — 只取输入序列长度的 PE，广播到 batch
        return x + self.pe[:, :x.size(1), :]              

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_k=64, d_ff=2048, dropout=0.1):
        super().__init__()
        # self.attention = MultiHeadAttention(d_model, n_heads, d_k)
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, d_ff),
        #     nn.ReLU(),
        #     nn.Linear(d_ff, d_model)
        # )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)  # 可选
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. attn_out, attn_weights = self.attention(x, mask)
        attn_out, attn_weights = self.attention(x, mask)
        # 2. x = self.norm1(x + self.dropout(attn_out))
        x = self.norm1(x + self.dropout(attn_out))
        # 3. ffn_out = self.ffn(x)
        ffn_out = self.ffn(x)
        # 4. x = self.norm2(x + self.dropout(ffn_out))
        x = self.norm2(x+self.dropout(ffn_out))
        # 5. return x, attn_weights
        return x, attn_weights
               


if __name__ == "__main__":
    B, L, D = 2, 10, 512
    n_heads, d_k = 8, 64

    x = torch.randn(B, L, D)

    mha = MultiHeadAttention(d_model=D, n_heads=n_heads, d_k=d_k)
    out, attn = mha(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Attn:   {attn.shape}")

    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print(f"Row sum check: {row_sums[0, 0, 0]:.6f}")

    mask = torch.tril(torch.ones(L, L))
    out_masked, attn_masked = mha(x, mask=mask)
    assert torch.all(attn_masked[:, :, 0, 5:] == 0)
    print("Causal mask check passed")

    # PE 测试
    pe = PositionalEncoding(d_model=512)
    x_pe = pe(x)
    assert x_pe.shape == x.shape, f"PE shape mismatch: {x_pe.shape}"
    # PE 第二个 batch 的向量应和第一个相同（PE 只依赖位置，不依赖 batch）
    assert torch.allclose(x_pe[0] - x[0], x_pe[1] - x[1], atol=1e-6), "PE not batch-consistent"
    print("PE tests passed")

    # EncoderBlock 测试
    block = EncoderBlock(d_model=D, n_heads=n_heads, d_k=d_k, d_ff=2048)
    out, attn = block(x)
    assert out.shape == x.shape, f"EncoderBlock output shape: {out.shape}"
    assert attn.shape == (B, n_heads, L, L), f"EncoderBlock attn shape: {attn.shape}"
    # 带 mask
    out_m, attn_m = block(x, mask=mask)
    assert out_m.shape == x.shape
    print("EncoderBlock tests passed")

    print("\nAll tests passed")