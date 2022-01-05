import torch
import torch.nn as nn
import math


# 2. 向量位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p = dropout)

        # （max_len, d_model）矩阵，保存每个位置的编码
        # 每个位置用一个d_model维度的向量表示其位置编码
        pe = torch.zeros(max_len, d_model)

        # (max_len) -> （max_len, 1）
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
        pe = pe.unsqueeze(0) # (max_len, d_model) -> (1, max_len, d_model)，为batch size留出位置

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        # 接受1.Embeddings的词嵌入结果x，
        #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        #例如，假设x是(30,10,512)的一个tensor，
        #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。

        return self.dropout(x)
        # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。