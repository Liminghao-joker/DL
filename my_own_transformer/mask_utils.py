import torch

# 创建填充掩码
def make_padding_mask(token_ids:torch.Tensor, pad_id:int) -> torch.Tensor:
    B, L = token_ids.shape
    padding_mask = (token_ids == pad_id) # [B, L]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    return padding_mask

# 创建因果掩码
def make_causal_mask(seq_len:int) -> torch.Tensor:
    """创建因果掩码，防止看到未来信息"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # 上三角为True
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
