# padding_mask: 形状 [B,1,1,L]；True 表示需要屏蔽。

# causal_mask: 上三角（不看未来），形状 [1,1,L,L]。

# 合并 mask 后在 logits 上 masked_fill(-inf) 再 softmax。
import torch
from torch import nn

class EBD(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        词嵌入层

        Args:
            vocab_size: 词汇表大小
            d_model: 嵌入维度
        """
        super(EBD, self).__init__()
        self.d_model = d_model

        # 词嵌入
        self.word_emb = nn.Embedding(vocab_size, d_model)

        #todo 位置编码(稍后添加)

        #todo Dropout(稍后添加)

    # X:[B, L]
    #todo 添加掩码
    def forward(self, X, padding_mask=None, causal_mask=None):
        """
        前向传播

        Args:
            x:token ids [B, L]
            padding_mask:
            casual_mask:
        
        Returns:
            embedded: 嵌入向量 [B, L, D]

        """
        B, L = X.shape
        word_embedded = self.word_emb(X)
        return word_embedded # [B, L, D]


# 创建填充掩码
def make_padding_mask(token_ids:torch.Tensor, pad_id:int):
    B, L = token_ids.shape

    padding_mask = (token_ids == pad_id) # [B, L]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    return padding_mask




if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, H = 2, 6, 32, 4
    pad_id = 0
    assert D % H == 0

    # test make_causal_mask
    causal_mask = make_causal_mask(L)
    # assert causal_mask.shape == (1, 1, L, L)
    # assert causal_mask[0, 0, 0, 1] == 0 
    # assert causal_mask[0, 0, 1, 0] == 1
    print(causal_mask)

