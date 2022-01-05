import torch
import torch.nn as nn
from sublayer_connection import *
from utils import *

class DecoderLayer(nn.Module):
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size = d_model=512,
        # self_attn = one MultiHeadAttention object，目标语言序列的
        # src_attn = second MultiHeadAttention object, 目标语言序列
        # 和源语言序列之间的
        # feed_forward 一个全连接层
        # dropout = 0.1
        super(DecoderLayer, self).__init__()
        self.size = size # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        # 需要三个SublayerConnection, 分别在self.self_attn, self.src_attn, 和self.feed_forward的后边

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # (batch.size, sequence.len, 512) 
        # 来自源语言序列的Encoder之后的输出，作为memory
        # 供目标语言的序列检索匹配：（类似于alignment in SMT)
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过一个匿名函数，来实现目标序列的自注意力编码
        # 结果扔给sublayer[0]:SublayerConnection

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_attn))
        # 通过第二个匿名函数，来实现目标序列和源序列的注意力计算
        # 结果扔给sublayer[1]:SublayerConnection

        return self.sublayer[2](x, self.feed_forward)

