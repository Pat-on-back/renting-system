# 导入必要的模块
import numpy as np
from torch import nn
import torch
import math
from collections import Counter, OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import LambdaLR
from torchtext.vocab import vocab
from functools import partial
from torch.nn.utils.rnn import pad_sequence

# LayerNorm层的实现，也可以直接使用nn.LayerNorm
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 词嵌入层
# 将token在词典中的id映射为一个d_model维的向量
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: tokenlized sequence
        :return:
        """
        # 乘以一个较大的系数，放大词嵌入向量，
        # 希望与位置编码向量相加后，词嵌入向量本身的影响更大
        return self.embed(x) * math.sqrt(self.d_model)

  # 预先计算好所有可能的位置编码，然后直接查表就能得到

# 注意维度顺序
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X的形状为 (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

  # 多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, head_number, d_model):
        """
        :param head_number: 自注意力头的数量
        :param d_model: 隐藏层的维度
        """
        super().__init__()
        self.h = head_number
        self.d_model = d_model
        self.dk = d_model // head_number
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(0.1)

    def head_split(self, tensor, batch_size):
        # 将(batch_size, seq_len, d_model) reshape成 (batch_size, seq_len, h, d_model//h)
        # 然后再转置第1和第2个维度，变成(batch_size, h, seq_len, d_model/h)
        return tensor.view(batch_size, -1, self.h, self.dk).transpose(1, 2)

    def head_concat(self, similarity, batch_szie):
        # 恢复计算注意力之前的形状
        return similarity.transpose(1, 2).contiguous() \
            .view(batch_szie, -1, self.d_model)

    def cal_attention(self, q, k, v, mask=None):
        """
        论文中的公式 Attention(K,Q,V) = softmax(Q@(K^T)/dk**0.5)@V
        ^T 表示矩阵转置
        @ 表示矩阵乘法
        """
        similarity = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # 将mask为0的位置填充为绝对值非常大的负数
            # 这样经过softmax后，其对应的权重就会非常接近0, 从而起到掩码的效果
            similarity = similarity.masked_fill(mask == 0, -1e9)
        similarity = self.softmax(similarity)
        similarity = self.dropout(similarity)

        output = torch.matmul(similarity, v)
        return output

    def forward(self, q, k, v, mask=None):
        """
        q,k,v即自注意力公式中的Q,K,V，mask表示掩码
        """
        batch_size, seq_length, d = q.size()
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        # 分成多个头
        q = self.head_split(q, batch_size)
        k = self.head_split(k, batch_size)
        v = self.head_split(v, batch_size)
        similarity = self.cal_attention(q, k, v, mask)
        # 合并多个头的结果
        similarity = self.head_concat(similarity, batch_size)

        # 再使用一个线性层， 投影一次
        output = self.output(similarity)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff, dropout=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dff, d_model)
        if dropout is not None:
            self.dropout = nn.Dropout(0.1)
        else:
            self.dropout = None

    def forward(self, x):
        """
        :param x: 来自多头自注意力层的输出
        :return:
        """
        x = self.fc1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.fc2(x)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, head_number, d_model, d_ff, dropout=0.1):
        super().__init__()

        # mha
        self.mha = MultiHeadAttention(head_number, d_model)
        self.norm1 = LayerNorm(d_model)

        # mlp
        self.mlp = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        
        y = self.dropout1(self.mha(x2, x2, x2, mask))
        # 注意残差连接是和norm之前的输入相加，norm之后的不在一个数量级
        y = y + x
        
        y2 = self.norm2(y)
        y2 = self.dropout2(self.mlp(y2))
        y2 = y + y2

        return y2

  class Encoder(nn.Module):
    def __init__(self, stack=6, multi_head=8, d_model=512, d_ff=2048):
        """
        :param stack: 堆叠多少个编码器层
        :param multi_head: 多头注意力头的数量
        :param d_model: 隐藏层的维度
        """
        super().__init__()
        self.encoder_stack = []
        for i in range(stack):
            encoder_layer = EncoderLayer(multi_head, d_model, d_ff)
            self.encoder_stack.append(encoder_layer)
        self.encoder = nn.ModuleList(self.encoder_stack)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask)
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, head_number, d_model, d_ff, dropout=0.1):
        super().__init__()
        # shifted right self attention layer
        self.mha1 = MultiHeadAttention(head_number, d_model)

        # cross attention
        self.mha2 = MultiHeadAttention(head_number, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.mlp = FeedForwardNetwork(d_model, d_ff, 0.1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask, tgt_mask):
        # 注意第一个注意力层的qkv都是同一个
        x2 = self.norm1(q)
        y = self.mha1(x2, x2, x2, tgt_mask)
        y = self.dropout1(y)
        
        # 注意残差连接是和norm之前的输入相加，norm之后的不在一个数量级
        y = y + q
        
        y2 = self.norm2(y)
        
        # 第二个自注意力层的k和v是encoder的输出
        y2 = self.mha2(y2, k, v, src_mask)
        y2 = self.dropout2(y2)
        y2 = y + y2
        
        y3 = self.norm3(y2)
        y3 = self.mlp(y3)
        y3 = self.dropout3(y3)
        y3 = y2 + y3

        return y3

  class Decoder(nn.Module):
    def __init__(self, stack=6, head_number=8, d_model=512, d_ff=2048):
        super().__init__()
        self.decoder_stack = []
        for i in range(stack):
            self.decoder_stack.append(DecoderLayer(head_number, d_model, d_ff))
        self.decoder_stack = nn.ModuleList(self.decoder_stack)
        self.norm = LayerNorm(d_model)

    def forward(self, x, output_from_encoder, src_mask, tgt_mask):
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x, output_from_encoder, output_from_encoder, src_mask, tgt_mask)
        x = self.norm(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self,
                 src_voc_size,
                 target_voc_size,
                 stack_number=6,
                 d_model=512,
                 h=8,
                 d_ff=2048):
        super().__init__()
        self.input_embedding_layer = Embedder(src_voc_size, d_model)
        self.input_pe = PositionalEncoding(d_model)

        self.output_embedding_layer = Embedder(target_voc_size, d_model)
        self.output_pe = PositionalEncoding(d_model)

        self.encoder = Encoder(stack_number, h, d_model, d_ff)
        self.decoder = Decoder(stack_number, h, d_model, d_ff)
        self.final_output = nn.Linear(d_model, target_voc_size)

    def encode(self, src, src_mask):
        x = self.input_embedding_layer(src)
        x = self.input_pe(x)
        output_from_encoder = self.encoder(x, src_mask)
        return output_from_encoder

    def decode(self, output_from_encoder, shifted_right, src_mask, tgt_mask):
        shifted_right = self.output_embedding_layer(shifted_right)
        shifted_right = self.output_pe(shifted_right)

        decoder_output = self.decoder(shifted_right, output_from_encoder, src_mask, tgt_mask)

        output = self.final_output(decoder_output)
        return output

    def forward(self, x, shifted_right, src_mask, tgt_mask):
        x = self.input_embedding_layer(x)
        x = self.input_pe(x)

        output_from_encoder = self.encoder(x, src_mask)

        shifted_right = self.output_embedding_layer(shifted_right)
        shifted_right = self.output_pe(shifted_right)

        decoder_output = self.decoder(shifted_right, output_from_encoder, src_mask, tgt_mask)

        output = log_softmax(self.final_output(decoder_output), dim=-1)

        return output

def build_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    model = TransformerModel(src_vocab, tgt_vocab, N, d_model, h, d_ff)
    # 权重初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def basic_test():
    a = np.random.randint(0, 100, size=(1, 10)).astype(np.int32)
    b = torch.from_numpy(a)

    a2 = np.random.randint(0, 100, size=(1, 12)).astype(np.int32)
    b2 = torch.from_numpy(a2)
    model = build_model(100, 120)
    print(model)
    output = model(b, b2, None, None)
    print(output.shape)


from datasets import load_dataset
if __name__ == '__main__':
    dataset = load_dataset('ted_hrlr', name='pt_to_en',trust_remote_code=True)
    
    train_data = dataset['train']['translation']
    train_data = dataset['train']['translation']
    
    print(type(train_data))
    print("训练集数量为:{}\n".format(len(train_data)))
    for data in train_data[:1]:
        en_seq = data['en']
        pt_seq = data['pt']
        print('英语句子为:\n{}'.format(en_seq))
        print('葡萄牙语句子为:\n{}'.format(pt_seq))
        print('\n')
    
    def build_vocab(seqs, tokenizer, k=None):
        counter = Counter()
        for seq in seqs:
            if k is not None:
                seq = seq[k]
            token = tokenizer(seq)
            counter.update(token)
        sorted_by_freq_tuples = sorted(counter.items(),
                                       key=lambda x: x[1],
                                       reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        # 指定特殊字符，特殊字符会被放在字典的起始未知，比如'<pad>'的索引就是0
        voc = vocab(ordered_dict, specials=['<pad>', '<unk>', '<bos>', '<eos>'])
        return voc


    def generate_batch(data_batch):
        pt_batch, en_input_batch, en_label_batch = [], [], []
        for pt, eni, enl in data_batch:
            pt_batch.append(pt)
            en_input_batch.append(eni)
            en_label_batch.append(enl)

        pt_batch = pad_sequence(pt_batch, padding_value=0, batch_first=True)
        # 使用0来补齐
        en_input_batch = pad_sequence(en_input_batch, padding_value=0, batch_first=True)
        en_label_batch = pad_sequence(en_label_batch, padding_value=0, batch_first=True)

        return pt_batch, en_input_batch, en_label_batch


    def seq_to_index(seq, tokenizer, voc):
        # 添加起始字符和结束字符
        indexs = [voc['<bos>']] + [voc[token] for token in tokenizer(seq)] + [voc['<eos>']]
        return torch.tensor(indexs, dtype=torch.int64)


    class TranslationDataset(Dataset):
        def __init__(self, dataset, en_voc, pt_voc, en_tokenizer, pt_tokenizer):
            self.dataset = dataset
            self.en_tokenizer = en_tokenizer
            self.pt_tokenizer = pt_tokenizer
            self.en_voc = en_voc
            self.pt_voc = pt_voc

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            en_seq, pt_seq = self.dataset[item]['en'], self.dataset[item]['pt']

            # seq to token list. start and end token added
            en_tensor = seq_to_index(en_seq, self.en_tokenizer, self.en_voc)
            pt_tensor = seq_to_index(pt_seq, self.pt_tokenizer, self.pt_voc)

            ground_truth = en_tensor[1:]  # drop the start token
            shifted_right = en_tensor[:-1]  # drop the end token
            return pt_tensor, shifted_right, ground_truth

        
    def build_dataloader(batch_size, cache_dir=None):
        # 读取《葡萄牙语-英语》翻译数据集
        dataset = load_dataset('ted_hrlr', name='pt_to_en', cache_dir=cache_dir)

        # 获取训练集切分和验证集切分
        train_data = dataset['train']['translation']
        val_data = dataset['validation']['translation']

        # 分词器
        en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        pt_tokenizer = get_tokenizer('spacy', language='pt_core_news_sm')

        # 构建字典索引
        en_voc = build_vocab(train_data+val_data, en_tokenizer, k='en')
        pt_voc = build_vocab(train_data+val_data, pt_tokenizer, k='pt')

        # 创建dataloader
        train_ds = TranslationDataset(train_data, en_voc, pt_voc, en_tokenizer, pt_tokenizer)
        val_ds = TranslationDataset(val_data, en_voc, pt_voc, en_tokenizer, pt_tokenizer)
        train_iter = DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, collate_fn=generate_batch)
        # 将验证集的batch_size设置为1
        valid_iter = DataLoader(val_ds, batch_size=1,
                                shuffle=False, collate_fn=generate_batch)
        return train_iter, valid_iter, len(en_voc), len(pt_voc), en_voc


    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0


    # 创建mask
    def create_mask(src, tgt, pad=0):
        src_mask = (src != pad).unsqueeze(-2)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return src_mask, tgt_mask

    GLOBAL_STEP = 1
    device='cuda'
    def train_one_step(model, loss_fn, batch_data, 
                    optimizer, lr_scheduler=None, 
                    log_writter=None,
                    global_step=None):
        # 切换到训练模式
        model.train()
        
        # x1为源语言序列，x2为shifted right,label表示目标语言序列，也就是标签
        x1, x2, label = batch_data

        # 创建掩码
        # src掩码的作用是为了忽略padding的字符
        # tgt掩码的作用是自回归时候遮蔽当前位置之后的字符
        src_mask, tgt_mask = create_mask(x1, x2)

        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # 前向获得输出
        pred = model(x1, x2, src_mask, tgt_mask)

        label = label.to(torch.long)
        pred = pred.contiguous().view(-1, pred.size(-1))
        label = label.contiguous().view(-1, )
        
        # 计算损失
        loss = loss_fn(pred, label)
        
        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        if log_writter is not None:
            log_writter.add_scalar('train_loss', loss, global_step)
        return loss

    def rate(step, model_size=512, warmup=4000):
        # 学习率调度
        if step == 0:
            step = 1
        return (
                model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def train():
        # 超参数设置
        batch_size = 48
        epochs = 60
        checkpoint_interval = 1

        # 创建数据输入管道
        training_data, val_data, tgt_voc_size, src_voc_size, tgt_voc = build_dataloader(batch_size)

        # 构建模型
        model = build_model(src_voc_size, tgt_voc_size, N=6, d_ff=1024)
        model.to(device)

        # 优化器，注意此处的lr设置比较大，作为一个基础学习率，经过调度器缩放后，数量级会减小
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )

        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step)
        )

        # 交叉熵损失函数，使用label smoothing， 忽略标签中的0, 0表示补齐占位符
        loss_fn = partial(F.cross_entropy, ignore_index=0, label_smoothing=0.1)

        # logger = SummaryWriter('./logs/')
        logger = None
        # 训练Loop
        global GLOBAL_STEP
        for epoch in range(1, epochs + 1):
            for step, batch_data in enumerate(training_data):
                # 在一个batch上训练
                loss = train_one_step(model, loss_fn, batch_data, optimizer, lr_scheduler, log_writter=logger,
                                    global_step=GLOBAL_STEP)
                GLOBAL_STEP += 1
                if step % 200 == 0:
                    print("epoch {}/{}  step:{}  loss:{:.3f}".format(epoch, epochs, step, loss))
            if epoch % checkpoint_interval == 0:
                # 周期性保存最新的模型
                torch.save(model.state_dict(), 'latest_ckpt.pth')
        
        if logger is not None:
            logger.flush()
            logger.close()
        torch.save(model.state_dict(), 'transformer.pth')

    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        memory = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(
                memory, ys, src_mask, subsequent_mask(ys.size(1)).type_as(src.data)
            )[0]
            prob = F.softmax(out, dim=-1)
            # 每次取预测的最后一个字符
            next_word = torch.argmax(prob, dim=-1)[-1].data
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys

    def token2text(tokens, voc_itos, eos=None):
        res = []
        for i, token in enumerate(tokens):
            c = voc_itos[token]
            if eos is not None:
                if c == eos:
                    break
            res.append(c)
        return ' '.join(res)

    def test():
        device = 'cuda'
        # dataloader
        batch_size = 1
        training_data, val_data, tgt_voc_size, src_voc_size, tgt_voc = build_dataloader(batch_size)
        # 构建模型
        model = build_model(src_voc_size, tgt_voc_size, N=6, d_ff=1024)
        # 加载训练好的模型
        model.load_state_dict(torch.load('transformer.pth'))
        model.to(device)
        
        # 切换到测试模式
        model.eval()
        max_len = 128
        itos = tgt_voc.get_itos()
        for _ in range(3):
            val_data = iter(val_data)
            batch_data = next(val_data)
            x1, x2, label = batch_data
            src_mask, tgt_mask = create_mask(x1, x2)
            x1 = x1.to(device)
            src_mask = src_mask.to(device)
            
            # 贪心解码
            pred_indexs = greedy_decode(model, x1, src_mask, max_len, 2)[0]
            
            # 将token ID转换成对应的字符
            pred_text = token2text(pred_indexs, itos, eos='<eos>')
            label = token2text(label[0], itos, eos='<eos>')
            print("实际的英语句子为:")
            print(label)
            print("翻译得到的句子为:")
            print(pred_text)
            

        
  
  



  
  

  
