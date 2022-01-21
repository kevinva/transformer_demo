import torch
import torch.nn as nn


def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = random.randrange(len(sentences)), random.randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        # tokens_a第一句，tokens_b第二句
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        print('[make_batch] input_ids: ', input_ids)
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        print('[make_batch] segment_ids: ', input_ids)
        
        n_pred = min(max_pred, max(int(round(len(input_ids) * 0.15)), 1))
        print('[make_batch] n_pred: ', n_pred)
        # 整个序列中15%的token
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        print('[make_batch] cand_maked_pos: ', cand_maked_pos)
        # 序列中可以添加[MASK]的位置
        
        shuffle(cand_maked_pos)
        print('[make_batch] after shuffle cand_maked_pos: ', cand_maked_pos)
        
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            elif random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]
        print('[make_batch] masked input_ids: ', input_ids)
        
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        
        if max_pred > n_pred:  # hoho: 这是啥？
            n_pad = mas_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            maksed_pos.extend([0] * n_pad)
        
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # 是下一句
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, mask_pos, False]) # 不是下一句
            negative += 1

    return batch

def get_attn_pad_mask(seq_q, seq_k):  # seq_q: query, seq_k: key
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # batch_size x 1 x len_k(=len_q), one is masking
    print('[get_attn_pad_mask] pad_attn_mask: ', pad_attn_mask.size())  
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
    # 使用expand将原本batch_size x 1 x len_k的size 扩展成# batch_size x len_q x len_k

class Embedding(nn.Module):

    def __init__(self):
        super(Embedding, self).__init__()

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_ebmed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype = torch.long)
        pos = pos.unsqueeze(0).expand_as(x) # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_ebmed(seg)
        return self.norm(embedding)


class EncoderLayer(nn.Module):
    
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # d_k为Key向量维度，d_v为Value向量的维度
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # hoho: attn_mask repeat前后变化？ 
        ## 答：对张量对应维度进行复制（如：参数有两个时，第1个参数表示在行方向上复制的次数，第二参数表示在列方向上复制的次数） 

        print('[MultiHeadAttention] 1. attn_mask: ', attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        print('[MultiHeadAttention] 2. attn_mask: ', attn_mask)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)

        return nn.LayerNorm(d_model)(output + residual), attn  # output: [batch_size x len_q x d_model]

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(score) # -1表示倒数第一个维度
        print('[ScaledDotProductAttention] attn: ', attn.size())
        print('[ScaledDotProductAttention] V: ', V.size())

        context = torch.matmul(attn, V)
        return score, context, attn


class BERT(nn.Module):
    
    def __init__(self):
        super(BERT, self).__init__()

        self.embedding = Embedding()   # 注意：这个是自定义的Embedding类
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias = False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    
    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.ebmedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, iput_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        # hoho: gather是什么鬼?
        # 答： 从output中,按维度dim=1,取出masked_pos所示各个索引的值 （可参考：https://zhuanlan.zhihu.com/p/352877584）
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias   # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf

# Gaussian Error Linear Unit
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))  
    # hoho: erf什么鬼？
    # 答：一个误差函数（可参考：https://www.cnblogs.com/htj10/p/8621771.html）


if __name__ == '__main__':
    tt = torch.tensor([1, 2, 3, 4, 5, 6])
    # print(tt.size())
    # print(tt.expand(2, 6, 6))

    print(tt.repeat(4, 2, 3, 2))