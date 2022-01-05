import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# 计算注意力：Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V 
def attention(query, key, value, mask = None, dropout = None):
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64), (30, 8, 11, 64)，
    # 例如30是batch.size，即当前batch中有多少个序列；
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

# 3. 
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
    
    def forward(self, query, key, value, mask = None):
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

if __name__ == '__main__':
    x = torch.tensor([[
                          [1, 2, 3, 3], 
                          [4, 5, 6, 6]
                       ], 
                      [
                          [2, 3, 4, 4], 
                          [5, 6, 7, 7]
                        ], 
                      [
                          [3, 4, 5, 5], 
                          [6, 7, 8, 8]
                        ]],
                        dtype = torch.float32)
    print('input sizer: ', x.size())

    attn = MultiHeadedAttention(4, 4)
    # out = attn(x, x, x)

    f= lambda x: attn(x, x, x)
    print('out: ', f)