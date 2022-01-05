import copy
import torch
import torch.nn as nn
import numpy as np

# 深copy
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    # triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0 
    # 上三角中元素为1.  

    return torch.from_numpy(subsequent_mask) == 0


if __name__ == '__main__':
    print(subsequent_mask(5))