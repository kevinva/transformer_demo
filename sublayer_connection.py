import torch
import torch.nn as nn
from layer_norm import LayerNorm

# 4.SubLayerConnection类主要实现两个功能，残差Add以及Norm（使用上面的LayerNorm类）：
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x is alike (batch.size, sequence.len, 512)
        # sublayer是一个具体的MultiHeadAttention
        #或者PositionwiseFeedForward对象
        return x + self.dropout(sublayer(self.norm(x)))
        # x (30, 10, 512) -> norm (LayerNorm) -> (30, 10, 512)
        # -> sublayer (MultiHeadAttention or PositionwiseFeedForward)
        # -> (30, 10, 512) -> dropout -> (30, 10, 512)
        
        # 然后输入的x（没有走sublayer) + 上面的结果，即实现了残差相加的功能