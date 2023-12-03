import torch

def subsequent_mask(size):
    "Mask out subsequent positions"
    attention_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None