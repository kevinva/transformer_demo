import numpy as np
import torch.nn as nn
import torch

norm = nn.BatchNorm1d(4, affine=False)
inputs = torch.FloatTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print('inputs: ', inputs)
boutput = norm(inputs)
print('b output: ', boutput)

b_mean = torch.mean(inputs, dim=0)
b_std = torch.std(inputs, dim=0)
print('mean: ', b_mean)
print('std: ', b_std)
print('manual: ', (inputs - b_mean) / b_std)

lnorm = nn.LayerNorm(4)
loutput = lnorm(inputs)
print('l output: ', loutput)