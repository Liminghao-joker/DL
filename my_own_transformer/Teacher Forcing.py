import torch

# 目标右移
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

    #! nn.Embedding 要求输入的是 LongTensor
    # 解码器输入
    # 在序列开头添加 <bos>,并移除最后一个token
    bos_tokens = torch.full((B, 1), bos_id if bos_id is not None else 0, dtype=torch.long, device=tgt_ids.device)
    decoder_input = torch.cat([bos_tokens, tgt_ids[:, :-1]], dim=1)

    # 解码器目标
    # 在序列结尾添加 <eos>
    eos_tokens = torch.full((B, 1), eos_id if eos_id is not None else 0, dtype=torch.long, device=tgt_ids.device)
    decoder_target = torch.cat([tgt_ids, eos_tokens], dim=1)

    return decoder_input, decoder_target

if __name__ == "__main__":
        target = torch.tensor([[5, 6, 7, 8, 0, 0], [1, 2, 3, 0, 0, 0]])  # 带填充的序列
        pad_id, bos_id, eos_id = 0, 101, 102
        decoder_input, decoder_target = shift_right(target, bos_id, eos_id)

        print("原始目标序列 (0=填充):")
        print(target)
        print("右移后的Decoder输入:")
        print(decoder_input)
        print("Decoder目标输出:")
        print(decoder_target)
        
        # 解释
        print("\n解释:")
        print("1. 原始序列右移一位，开头添加<SOS>")
        print("2. 目标序列在有效序列结尾添加<EOS>，保持填充不变")
        print("3. 这样模型在训练时不会看到'未来'信息")