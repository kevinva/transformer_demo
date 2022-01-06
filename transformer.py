import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from utils import *
from multi_head_attention import *
from positionwise_feed_forward import *
from positional_encoding import *
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from embedding import Embeddings
from layer_norm import *


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 深copy，N=6，
        self.norm = LayerNorm(layer.size)
        # 定义一个LayerNorm，layer.size=d_model=512
        # 其中有两个可训练参数a_2和b_2

    def forward(self, x, mask):
        # x is alike (30, 10, 512)
        # (batch.size, sequence.len, d_model)
        # mask是类似于(batch.size, 10, 10)的矩阵
        for layer in self.layers:
            x = layer(x, mask)
            # 进行六次EncoderLayer操作
        return self.norm(x)
        # 最后做一次LayerNorm，最后的输出也是(30, 10, 512) shape


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # 执行六次DecoderLayer
        return self.norm(x)
        # 执行一次LayerNorm


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        # d_model=512
        # vocab = 目标语言词表大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)  # 注意：这里只是做softmax运算后，再做一次log运行，并不是交叉熵
        # x 类似于 (batch.size, sequence.length, 512)
        # -> proj 全连接层 (30, 10, trg_vocab_size) = logits
        # 对最后一个维度执行log_soft_max
        # 得到(30, 10, trg_vocab_size)


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        # 对源语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # 对目标语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor


def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),  ### hoho: 这里为啥要包一层nn.Sequential?
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    # x = torch.tensor([[
    #                       [1, 2, 3], 
    #                       [4, 5, 6]
    #                    ], 
    #                   [
    #                       [2, 3, 4], 
    #                       [5, 6, 7]
    #                     ], 
    #                   [
    #                       [3, 4, 5], 
    #                       [6, 7, 8]
    #                     ]],
    #                     dtype = torch.float32)
    # print(x.size())
 
    # print('softmax: ', F.softmax(x, dim = -1))
    # print('log_softmax: ', F.log_softmax(x, dim = -1))