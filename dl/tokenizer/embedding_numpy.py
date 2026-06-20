import numpy as np

class Embedding():
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, token_ids):

        return self.weight[token_ids]


embed = Embedding(vocab_size=16, d_model=8)
# 喂 BPE encode 的输出
ids = [[14, 10, 0], [15, 11, 13]]  # "low newest" 的 BPE 编码
out = embed.forward(ids)
print(out.shape)  # 期望 (2, 3, 8)