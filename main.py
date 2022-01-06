from transformer import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *

def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size = (batch, 10)))
    data[:, 0] = 1
    src = Variable(data, requires_grad = False)
    tgt = Variable(data, requires_grad = False)
    yield Batch(src, tgt, 0)

class SimpleLossCompute:

    def __init__(self, generator, criterion, opt = None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # e.g., x为(2,3,8), batch.size=2, seq.len=3, d_model=8
        # y = tensor([[4, 2, 1],
        #[4, 4, 4]], dtype=torch.int32)

        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm.item()
        # 变形后，x类似于(batch.size*seq.len, target.vocab.size)
        # y为(target.vocab.size)
        # 然后调用LabelSmooth来计算loss
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        
        return loss.data.item() * norm.item()


def run_epoch(aepoch, data_iter, model, loss_compute):

    "Standard Training and Logging Function"
    # data_iter = 所有数据的打包
    # model = EncoderDecoder 对象
    # loss_compute = SimpleLossCompute对象
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # 对每个batch循环
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        # 使用目前的model，对batch.src+batch.trg进行forward
                            
        # e.g.,
        # batch.src (2,4) = tensor([[1, 4, 2, 1],
        # [1, 4, 4, 4]], dtype=torch.int32)
        
        # batch.trg (2,3) = tensor([[1, 4, 2],
        # [1, 4, 4]], dtype=torch.int32)
        
        # batch.src_mask (2,1,4) = tensor([[[1, 1, 1, 1]],
        # [[1, 1, 1, 1]]], dtype=torch.uint8)
        
        # batch.trg_mask (2,3,3) = tensor([[[1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]],

        #[[1, 0, 0],
        # [1, 1, 0],
         #[1, 1, 1]]], dtype=torch.uint8)
         
        # and out (2,3,8):
        # out = tensor([[[-0.4749, -0.4887,  0.1245, -0.4042,  0.5301,  
        #   1.7662, -1.6224, 0.5694],
        # [ 0.4683, -0.7813,  0.2845,  0.4464, -0.3088, -0.1751, -1.6643,
        #   1.7303],
         #[-1.1600, -0.2348,  1.0631,  1.3192, -0.9453,  0.3538,  0.7051...                 
        
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        # out和trg_y计算Loss
        # ntokens = 6 (trg_y中非'<blank>'的token的个数)
        # 注意，这里是token,不是unique word
        # 例如[ [ [1, 2, 3], [2,3,4] ]中有6个token,而只有4个unique word
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            "attention here 这里隐藏一个bug"
            #print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            #        (i, loss / batch.ntokens, tokens / elapsed))
            print ('epoch step: {}:{} Loss: {}/{}, tokens per sec: {}/{}'
                    .format(aepoch, i, loss, batch.ntokens, 
                    tokens, elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask) 
    # 源语言的一个batch
    # 执行encode编码工作，得到memory 
    # shape=(batch.size, src.seq.len, d_model)
    
    # src = (1,4), batch.size=1, seq.len=4
    # src_mask = (1,1,4) with all ones
    # start_symbol=1
    
    print ('memory={}, memory.shape={}'.format(memory, 
        memory.shape))
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 最初ys=[[1]], size=(1,1); 这里start_symbol=1
    print ('ys={}, ys.shape={}'.format(ys, ys.shape))
    for i in range(max_len-1): # max_len = 5
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        # memory, (1, 4, 8), 1=batch.size, 4=src.seq.len, 8=d_model
        # src_mask = (1,1,4) with all ones
        # out, (1, 1, 8), 1=batch.size, 1=seq.len, 8=d_model                             
        print ('out={}, out.shape={}'.format(out, out.shape))
        prob = model.generator(out[:, -1]) 
        # pick the right-most word
        # (1=batch.size,8) -> generator -> prob=(1,5) 5=trg.vocab.size
        # -1 for ? only look at the final (out) word's vector
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        # word id of "next_word"
        ys = torch.cat([ys, 
          torch.ones(1, 1).type_as(src.data).fill_(next_word)], 
          dim=1)
        # ys is in shape of (1,2) now, i.e., 2 words in current seq
    return ys


if __name__ == '__main__':
    # Train the simple copy task.
    V = 5 # here V is the vocab size of source and target languages (sequences)
    criterion = LabelSmoothing(size=V, 
        padding_idx=0, smoothing=0.01) # 创建损失函数计算对象
        
    model = make_model(V, V, N=2, d_model=8, d_ff=16, h=2) 
    # EncoderDecoder对象构造
    '''
    in make_model: src_vocab_size=11, tgt_vocab_size=11, 
        N=2, d_model=512, d_ff=2048, h=8, dropout=0.1
    '''

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), 
            lr=0, betas=(0.9, 0.98), eps=1e-9))
    # 模型最优化算法的对象


    lossfun = SimpleLossCompute(model.generator, 
        criterion, model_opt)
    if True:
        print ('start model training...')
        for epoch in range(10):
            print ('epoch={}, training...'.format(epoch))
            model.train() # set the model into "train" mode
            # 设置模型进入训练模式
            
            #lossfun = SimpleLossCompute(model.generator, 
            #    criterion, model_opt) # 不需要在这里定义lossfun
            
            run_epoch(epoch, data_gen(V, 4, 2, 2), model, lossfun)
            # 重新构造一批数据，并执行训练
            
            model.eval() # 模型进入evaluation模式 (dropout，反向传播无效）
            print ('evaluating...')
            print(run_epoch(epoch, data_gen(V, 4, 2, 2), model, 
                            SimpleLossCompute(model.generator, 
                            criterion, None)))
            # 这里None表示优化函数为None，所以不进行参数更新 

    if True:
        model.eval()
        src = Variable(torch.LongTensor([[1,2,3,4]]))
        src_mask = Variable(torch.ones(1, 1, 4))
        print(greedy_decode(model, src, src_mask, max_len=5, 
            start_symbol=1))