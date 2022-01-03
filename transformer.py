import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

# 1. Embeddings
class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = nn.Embeddings(vocba, d_model)
        self.d_model = d_model  # d_model: 模型大小

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


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


def attention(query, key, value, mask = None, dropout = None):
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64), (30, 8, 11, 64)，
    # 例如30是batch.size，即当前batch中有多少一个序列；
    # 8=head.num，注意力头的个数；
    # 10=目标序列中词的个数，64是每个词对应的向量表示；
    # 11=源语言序列传过来的memory中，当前序列的词的个数，
    # 64是每个词对应的向量表示。
    # 类似于，这里假定query来自target language sequence；
    # key和value都来自source language sequence.

    d_k = query.size(-1)  # 64
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，
    #（注意是最后两个维度相乘）得到(30,8,10,11)，
    #代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    #然后除以sqrt(d_k)=8，防止过大的亲密度。
    #这里的scores的shape是(30, 8, 10, 11)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
        #然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视

    p_attn = F.softmax(scores, dim = -1)
    #对scores的最后一个维度执行softmax，得到的还是一个tensor, (30, 8, 10, 11)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        #定义四个Linear networks, 每个的大小是(512, 512)的，
        #每个Linear network里面有两类可训练参数，Weights，
        #其大小为512*512，以及biases，其大小为512=d_model。

        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, query, key, value, mask=None):
        # 注意，输入query的形状类似于(30, 10, 512)，
        # key.size() ~ (30, 11, 512), 
        #以及value.size() ~ (30, 11, 512)

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 这里是前三个Linear Networks的具体应用，(每个linear分别去query, key, value做zip)
        # 例如query=(30,10, 512) -> Linear network -> (30, 10, 512) 
        # -> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        # ，其他的key和value也是类似地，从(30, 11, 512) -> (30, 8, 11, 64)。

        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)
        #调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)；
        #attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)

        # 在使用transpose()进行转置操作时，pytorch并不会创建新的、转置后的tensor，而是修改了tensor中的一些属性。
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # x ~ (30, 8, 10, 64) -> transpose(1,2) -> 
        # (30, 10, 8, 64) -> contiguous() and view -> 
        # (30, 10, 8*64) = (30, 10, 512)
        return self.linears[-1](x)

# Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V 


if __name__ == '__main__':
    testArr = np.random.randn(10, 3, 5)
    testTensor = torch.tensor(testArr, dtype=torch.float32)
    # testTensor = torch.randn(10, 3, 5)
    print(testTensor)

    testLinear = nn.Linear(5, 5)
    out = testLinear(testTensor)
    print(out.size())