import torch

def subsequent_mask(size):
    "Mask out subsequent positions"
    attention_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0