import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# 深copy
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    # triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0 
    # 上三角中元素为1.  

    return torch.from_numpy(subsequent_mask) == 0


class Batch:

    def __init__(self, src, trg = None, pad = 0):
        # src: 源语言序列，(batch.size, src.seq.len)
        # 二维tensor，第一维度是batch.size；第二个维度是源语言句子的长度
        # 例如：[ [2,1,3,4], [2,3,1,4] ]这样的二行四列的，
        # 1-4代表每个单词word的id
        
        # trg: 目标语言序列，默认为空，其shape和src类似
        # (batch.size, trg.seq.len)，
        #二维tensor，第一维度是batch.size；第二个维度是目标语言句子的长度
        # 例如trg=[ [2,1,3,4], [2,3,1,4] ] for a "copy network"
        # (输出序列和输入序列完全相同）
        
        # pad: 源语言和目标语言统一使用的 位置填充符号，'<blank>'
        # 所对应的id，这里默认为0
        # 例如，如果一个source sequence，长度不到4，则在右边补0
        # [1,2] -> [1,2,0,0]

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        # src = (batch.size, seq.len) -> != pad -> 
        # (batch.size, seq.len) -> usnqueeze ->
        # (batch.size, 1, seq.len) 相当于在倒数第二个维度扩展
        # e.g., src=[ [2,1,3,0], [2,3,1,4] ]对应的是
        # src_mask=[ [[1,1,1,0], [1,1,1,1]] ]

        if trg is not None:
            self.trg = trg[:, : -1]
            # trg 相当于目标序列的前N-1个单词的序列
            #（去掉了最后一个词）
            self.trg_y = trg[:, 1:]
            # trg_y 相当于目标序列的后N-1个单词的序列
            # (去掉了第一个词）
            # 目的是(src + trg) 来预测出来(trg_y)，
            # 这个在illustrated transformer中详细图示过。

            self.trg_mask = self.mask_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # 这里的tgt类似于：
        #[ [2,1,3], [2,3,1] ] （最初的输入目标序列，分别去掉了最后一个词
        # pad=0, '<blank>'的id编号
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 得到的tgt_mask类似于
        # tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        # shape=(2,1,3)

        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # 先看subsequent_mask, 其输入的是tgt.size(-1)=3
        # 这个函数的输出为= tensor([[
        # [1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]]], dtype=torch.uint8)
        # type_as 把这个tensor转成tgt_mask.data的type(也是torch.uint8)
        
        # 这样的话，&的两边的tensor分别是(2,1,3), (1,3,3);
        #tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        # and
        # tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.uint8)
        
        # (2,3,3)就是得到的tensor
        # tgt_mask.data = tensor([[
        # [1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]],

        #[[1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]]], dtype=torch.uint8)
        #
        # &左边每一个(1, 3)向量，共两个，分别去&右边的(1, 3, 3)向量做运算
        
        return tgt_mask

class NoamOpt:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None: 
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr = 0, betas = (0.9, 0.98), eps = 1e-9))

# 损失函数
class LabelSmoothing(nn.Module):

    def __init__(self, size, padding_idx, smoothing = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average = False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        "in real-world case: 真实情况下"
        # x的shape为(batch.size * seq.len, target.vocab.size)
        # target的shape是(batch.size * seq.len)
        
        # x=logits，(seq.len, target.vocab.size)
        # 每一行，代表一个位置的词
        # 类似于：假设seq.len=3, target.vocab.size=5
        # x中保存的是log(prob)
        # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])
        
        # target 类似于：
        # target = tensor([2, 1, 0])，torch.size=(3)
        
        assert x.size(1) == self.size # 目标语言词表大小
        true_dist = x.data.clone()
        # true_dist = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])
        
        true_dist.fill_(self.smoothing / (self.size - 2))
        # true_dist = tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        #[0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])
        
        # 注意，这里分母target.vocab.size-2是因为
        # (1) 最优值 0.6要占一个位置；
        # (2) 填充词 <blank> 要被排除在外
        # 所以被激活的目标语言词表大小就是self.size-2
        
        true_dist.scatter_(1, target.data.unsqueeze(1), 
          self.confidence)
        # target.data.unsqueeze(1) -> 
        # tensor([[2],
        #[1],
        #[0]]); shape=torch.Size([3, 1])  
        # self.confidence = 0.6
        
        # 根据target.data的指示，按照列优先(1)的原则，把0.6这个值
        # 填入true_dist: 因为target.data是2,1,0的内容，
        # 所以，0.6填入第0行的第2列（列号，行号都是0开始）
        # 0.6填入第1行的第1列
        # 0.6填入第2行的第0列：
        # true_dist = tensor([[0.1333, 0.1333, 0.6000, 0.1333, 0.1333],
        #[0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        #[0.6000, 0.1333, 0.1333, 0.1333, 0.1333]])
          
        true_dist[:, self.padding_idx] = 0
        # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        #[0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        #[0.0000, 0.1333, 0.1333, 0.1333, 0.1333]])
        # 设置true_dist这个tensor的第一列的值全为0
        # 因为这个是填充词'<blank>'所在的id位置，不应该计入
        # 目标词表。需要注意的是，true_dist的每一列，代表目标语言词表
        #中的一个词的id
        
        mask = torch.nonzero(target.data == self.padding_idx)
        # mask = tensor([[2]]), 也就是说，最后一个词 2,1,0中的0，
        # 因为是'<blank>'的id，所以通过上面的一步，把他们找出来
        
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            # 当target reference序列中有0这个'<blank>'的时候，则需要把
            # 这一行的值都清空。
            # 在一个batch里面的时候，可能两个序列长度不一，所以短的序列需要
            # pad '<blank>'来填充，所以会出现类似于(2,1,0)这样的情况
            # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
            # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
            # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        self.true_dist = true_dist
        return self.criterion(x, 
          Variable(true_dist, requires_grad=False))
          # 这一步就是调用KL loss来计算
          # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
          #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
          #[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])
          
          # true_dist=tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
          # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
          # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
          # 之间的loss了。细节可以参考我的那篇illustrated transformer


if __name__ == '__main__':
    # print(subsequent_mask(5))
    # tgt_mask = torch.tensor([[[1, 1, 1, 1]], 
    #                          [[1, 1, 1, 0]], 
    #                          [[1, 1, 1, 1]]])
    # subs = subsequent_mask(tgt_mask.size(-1))

    # print(tgt_mask.size())
    # print(subs.size())
    # print(tgt_mask & subs)

    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    predict = predict.masked_fill(predict == 0, 1e-9)
    v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
    plt.imshow(crit.true_dist)
    plt.show()