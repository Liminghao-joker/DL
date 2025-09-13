import torch
from torch import nn
# import seaborn as sns
import numpy as np

from mask_utils import make_causal_mask, make_padding_mask
# 计算注意力矩阵
def scaled_dot_attn(q, k, v, d_k, mask=None):
    # q,k,v: [B,H,L,d_k]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)  # [B,H,L,L]
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9) #! -torch.inf
    A = torch.softmax(scores, dim=-1)  # [B,H,L,L]
    out = torch.matmul(A, v)           # [B,H,L,d_k]
    return out, A

# 多头注意力机制
class MHA(torch.nn.Module):
    def __init__(self, d_model:int, n_head:int, bias:bool=False):
        super().__init__()
        assert d_model % n_head == 0 
        self.d_model, self.n_head, self.d_k = d_model, n_head, d_model // n_head
        self.W_q = torch.nn.Linear(d_model, d_model, bias=bias)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=bias)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=bias)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=bias)

    def attention(self, q, k, v, mask=None):
        """
        计算缩放点积注意力
        Args:
            q, k, v: [B,H,L,d_k] 查询、键、值张量
            mask: 掩码张量，形状可广播到 [B,H,L,L]
        Returns:
            out: [B,H,L,d_k] 注意力输出
            A: [B,H,L,L] 注意力权重矩阵
        """
        return scaled_dot_attn(q, k, v, self.d_k, mask)

    def forward(self, query, key=None, value=None, mask=None):
        """
        多头注意力前向传播
        Args:
            query: [B,L_q,D] 查询向量
            key: [B,L_k,D] 键向量，如果为None则使用query (自注意力)
            value: [B,L_v,D] 值向量，如果为None则使用query (自注意力) 
            mask: 掩码张量
        """
        # 自注意力：key和value都使用query
        if key is None:
            key = query
        if value is None:
            value = query
            
        assert query.dim() == 3 # [B,L_q,D]
        assert key.dim() == 3   # [B,L_k,D]
        assert value.dim() == 3 # [B,L_v,D]
        
        B, L_q, D = query.shape
        _, L_k, _ = key.shape
        _, L_v, _ = value.shape
        H, d_k = self.n_head, self.d_k
        
        # 确保key和value长度一致
        assert L_k == L_v, f"Key长度({L_k})和Value长度({L_v})必须一致"

        # 线性映射并拆头
        q = self.W_q(query).view(B, L_q, H, d_k).transpose(1, 2)  # [B,H,L_q,d_k]
        k = self.W_k(key).view(B, L_k, H, d_k).transpose(1, 2)    # [B,H,L_k,d_k]
        v = self.W_v(value).view(B, L_v, H, d_k).transpose(1, 2)  # [B,H,L_v,d_k]
        
        # 形状断言
        expected_q_shape = (B, H, L_q, d_k)
        expected_kv_shape = (B, H, L_k, d_k)
        assert q.shape == expected_q_shape, f"Q形状错误: 期望{expected_q_shape}, 实际{q.shape}"
        assert k.shape == expected_kv_shape, f"K形状错误: 期望{expected_kv_shape}, 实际{k.shape}"
        assert v.shape == expected_kv_shape, f"V形状错误: 期望{expected_kv_shape}, 实际{v.shape}"

        if mask is not None:
            # mask 期望形状可广播到 [B,H,L_q,L_k]
            assert mask.dim() in (3,4), f"mask维度应该是3或4，实际: {mask.dim()}"

        out, A = self.attention(q, k, v, mask)        # out: [B,H,L_q,d_k]

        assert out.shape == (B, H, L_q, d_k), f"注意力输出形状错误: 期望{(B,H,L_q,d_k)}, 实际{out.shape}"
        assert A.shape == (B, H, L_q, L_k), f"注意力权重形状错误: 期望{(B,H,L_q,L_k)}, 实际{A.shape}"

        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # [B,L_q,D]
        y = self.W_o(out)                               # [B,L_q,D]
        assert y.shape == (B, L_q, D), f"最终输出形状错误: 期望{(B,L_q,D)}, 实际{y.shape}"
        return y, A

    @staticmethod #? 为什么要添加静态装饰器？
    def combine_masks(*mask):
        """
        合并多个掩码
        Args:
            masks: 多个掩码张量
        Returns:
            combined_mask: 合并后的掩码
        """
        combined_mask = None
        # 依次传入 causal_mask, tgt_padding_mask
        for m in mask:
            if m is not None:
                if combined_mask is None:
                    combined_mask = m # 第一个非None的mask作为基础
                else:
                    combined_mask = combined_mask | m # 实现逻辑或操作
        return combined_mask


#* 嵌入层
class EBD(nn.Module):
    def __init__(self, vocab_size, d_model,max_len=2048, dropout_p = 0.1):
        super(EBD, self).__init__()
        self.d_model = d_model
        # 词嵌入
        self.word_emb = nn.Embedding(vocab_size, d_model)

        #? 位置编码
        pos_emb = sinusoidal_pe(max_len, d_model)
        self.register_buffer('pos_emb', pos_emb)  # 不作为模型参数更新

        #? Dropout (训练时再打开)
        self.dropout = nn.Dropout(dropout_p)

    # X:[B, L]
    #! 此时不添加掩码,padding_mask 应添加在计算注意力分数上面
    def forward(self, X, padding_mask=None, causal_mask=None):
        B, L = X.shape
        #? 词嵌入缩放
        out = self.word_emb(X) * (self.d_model ** 0.5)  # [B, L, D]
        out += self.pos_emb[:L].unsqueeze(0) # [B, L, D]
        # return self.dropout(out) # 训练
        return out # 测试

#* 绝对位置编码(sin/cos)
def sinusoidal_pe(seq_len:int, d_model:int):
    pos = torch.arange(seq_len).float().unsqueeze(1) # [L, 1]
    i = torch.arange(d_model//2).float().unsqueeze(0) # [1, D/2]
    angle = pos / torch.pow(10000, (2*i)/d_model) 
    # 创建一个空的tensor                           # [L,D/2]
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe  # [L,D]

# 逐位前馈网络
#! 为什么只需要一个激活函数？
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff * d_model, bias = False),
            nn.GELU(), #? 试试换成 ReLU
            nn.Linear(d_ff * d_model, d_model, bias = False)
        )

    def forward(self, x):
        return self.net(x)
    
#* Encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, n_head:int, d_ff:int, dropout_p:float=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_head, bias=False)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        # Pre-LN架构
        # 多头注意力子层
        norm_x = self.norm1(x) # 先归一化
        attn_out, attn_weights = self.mha(norm_x, mask=mask)
        x = x + self.dropout1(attn_out) # 残差连接

        # 前馈网络子层
        norm_x = self.norm2(x) # 先归一化
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout2(ffn_out) # 残差连接
        
        return x, attn_weights # [B, L, D] [B, L, L]

#* 编码器
class Encoder(nn.Module):
    """
    Transformer Encoder
    Args:
        vocab_size: 词汇表大小
        d_model: 嵌入维度
        n_head: 注意力头数
        n_layer: 编码器层数
        d_ff: 前馈网络维度
        max_len: 最大序列长度
        dropout: Dropout比例

    Return:
        最终的编码器输出和注意力权重
    """
    def __init__(self, vocab_size:int, d_model:int, n_head:int, n_layer:int, d_ff:int=4, max_len:int=2048, dropout=0.1):
        super().__init__()
        self.emb = EBD(vocab_size, d_model, max_len=max_len, dropout_p=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff=d_ff, dropout_p=dropout) 
            for _ in range(n_layer)
        ])
    
    def forward(self, token_ids, pad_id):
        x = self.emb(token_ids) # [B, L, D]
        padding_mask = make_padding_mask(token_ids, pad_id).to(x.device) # [B, 1, 1, L]

        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask=padding_mask)
            attention_weights_list.append(attention_weights) # [B, H, L, L]
        
        return x, attention_weights_list

#* Decoder层
class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, n_head:int, d_ff:int, dropout_p:float=0.1):
        super().__init__()
        # 自注意力层(因果掩码)
        self.self_attn = MHA(d_model, n_head, bias=False)
        # 交叉注意力(与encoder输出)
        self.cross_attn = MHA(d_model, n_head, bias=False)
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model) # 自注意力前
        self.norm2 = nn.LayerNorm(d_model) # 交叉注意力前
        self.norm3 = nn.LayerNorm(d_model) # FNN前

        # Dropout
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(self, x, encoder_output, causal_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        Args:
            x: 目标序列嵌入 [B, tgt_L, D]
            encoder_output: 编码器输出 [B, src_L, D]
            causal_mask: 因果掩码 [1, 1, tgt_L, tgt_L] - 用于自注意力
            src_padding_mask: 填充掩码 [B, 1, 1, src_L]
            tgt_padding_mask: 目标序列填充掩码 [B, 1, 1 ,tgt_L] - 用于自注意力
        """

        # 1. 自注意力子层(带因果掩码)
        norm_x = self.norm1(x)
        # 合并自注意力的掩码
        self_attn_mask = MHA.combine_masks(causal_mask, tgt_padding_mask)
        self_attn_out, self_attn_weights = self.self_attn(norm_x, mask=self_attn_mask)
        x = x + self.dropout1(self_attn_out)

        # 2. 交叉注意力子层
        norm_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=norm_x, 
            key=encoder_output, 
            value=encoder_output, 
            mask=src_padding_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # 3. 前馈网络子层
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout3(ffn_out)

        return x, self_attn_weights, cross_attn_weights

#* 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, n_head:int, n_layer:int, d_ff:int=4, max_len:int=2048, dropout=0.1):
        super().__init__()
        self.emb = EBD(vocab_size, d_model, max_len=max_len, dropout_p=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff=d_ff, dropout_p=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, tgt_ids, encoder_output, src_ids, src_pad_id, tgt_pad_id):
        """
        Args:
            tgt_ids: 目标序列token ids [B, tgt_L]
            encoder_output: 编码器输出 [B, src_L, D]
            src_ids: 源序列token ids [B, src_L]
            src_pad_id: 源序列PAD token id
            tgt_pad_id: 目标序列PAD token id
        """
        B, tgt_L = tgt_ids.shape
        src_L = encoder_output.shape[1]

        # 目标序列嵌入
        x = self.emb(tgt_ids)  # [B, tgt_L, D]

        # 创建掩码
        causal_mask = make_causal_mask(tgt_L).to(x.device)  # [1, 1, tgt_L, tgt_L]
        src_padding_mask = make_padding_mask(src_ids, src_pad_id).to(x.device)  # [B, 1, 1, src_L]
        tgt_padding_mask = make_padding_mask(tgt_ids, tgt_pad_id).to(x.device)  # [B, 1, 1, tgt_L]

        self_attn_weights_list = []
        cross_attn_weights_list = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, causal_mask, src_padding_mask, tgt_padding_mask
            )
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)

        return x, self_attn_weights_list, cross_attn_weights_list

# Teacher Forcing
def shift_right(tgt_ids:torch.Tensor, bos_id, eos_id=None) -> torch.Tensor:
    """
    目标序列右移
    Args:
        tgt_ids: [B, L]
        bos_id: 句子开始标记ID
        eos_id: 句子结束标记ID
    Returns:
        decoder_input: 解码器输入 [B, L]   
        decoder_target: 解码器目标 [B, L]
    """
    B, L = tgt_ids.shape

    # 解码器输入：在序列开头添加 <bos>，并移除最后一个token
    bos_tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=tgt_ids.device)
    decoder_input = torch.cat([bos_tokens, tgt_ids[:, :-1]], dim=1)  # [B, L]

    # 解码器目标：使用原始序列作为目标
    #! 原始序列以<eos>结尾
    decoder_target = tgt_ids  # [B, L]

    return decoder_input, decoder_target

#* 输出分类头
class Seq2SeqHead(nn.Module):
    """
    Transformer输出头:将隐藏状态映射到词汇表概率分布
    """
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.weight_tied = False  # 是否与嵌入层权重绑定

    def forward(self, dec_hidden):
        return self.proj(dec_hidden)
    
    #? tie with target embedding
    def tie_weights(self, embedding_layer):
        """权重绑定函数"""
        if hasattr(embedding_layer, 'weight'):
            if self.proj.weight.shape == embedding_layer.weight.shape:
                self.proj.weight = embedding_layer.weight
                self.weight_tied = True
                print(f"权重绑定成功：{self.proj.weight.shape}")
            else:
                raise ValueError(f"权重形状不匹配。output_layer:{self.proj.weight.shape} != embedding_layer:{embedding_layer.weight.shape}")
        else:
            raise ValueError("传入的对象没有weight属性")
        
#* transformer
class Transformer(nn.Module):
    """
    完整的Transformer模型: Encoder-Decoder架构
    """
    def __init__(self, src_vocab_size:int, tgt_vocab_size:int,d_model:int,
                n_head:int, n_enc_layer:int, n_dec_layer:int,d_ff:int=4,
                max_len:int=2048, dropout:float=0.1, weight_tying:bool=True,
                src_pad_id=0, tgt_pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # 编码器和解码器
        self.encoder = Encoder(src_vocab_size, d_model, n_head, n_enc_layer, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, n_dec_layer, d_ff, max_len, dropout)
        
        # 输出头
        self.output_head = Seq2SeqHead(d_model, tgt_vocab_size)

        # 权重绑定
        if weight_tying:
            self.output_head.tie_weights(self.decoder.emb.word_emb)

        # 参数初始化
        self.init_parameters()
    
    def init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if self.output_head.weight_tied and name == 'output_head.proj.weight':
                    continue
                nn.init.xavier_uniform_(p)


    def forward(self, src_ids, tgt_ids):
        
        # 使用Teacher Forcing 
        decoder_input, decoder_target = shift_right(tgt_ids, self.bos_id, self.eos_id)
        # 编码器
        encoder_output, enc_attn_weights = self.encoder(src_ids, self.src_pad_id)

        # 解码器
        decoder_output, dec_self_attn, dec_cross_attn = self.decoder(
            decoder_input, encoder_output, src_ids, self.src_pad_id, self.tgt_pad_id)

        # 输出头
        logits = self.output_head(decoder_output)

        return logits


if __name__ == "__main__":
    pass
