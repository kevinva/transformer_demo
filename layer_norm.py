import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()

        # a_2，b_2都是可训练参数向量，（512）
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        mean = x.mean(-1, keepdim = True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        std = x.std(-1, keepdim = True)
         # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0
        # 的时候的除法溢出


if __name__ == '__main__':
    x = torch.tensor([[
                          [1, 2, 3], 
                          [4, 5, 6]
                       ], 
                      [
                          [2, 3, 4], 
                          [5, 6, 7]
                        ], 
                      [
                          [3, 4, 5], 
                          [6, 7, 8]
                        ]],
                        dtype = torch.float32)
    print(x.size())
    
    mean = x.mean(dim = -1, keepdim = True)
    print(mean.size())

    std = x.std(-1, keepdim = True)
    print(std.size())

    print((x - mean) / std)