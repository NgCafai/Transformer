import time
import torch
from torch import nn
from models.utils import subsequent_mask

class Batch:
    """
    Object for holding a batch of src and target sentences for training,
    as well as constructing masks.
    """
    def __init__(self, src, tgt=None, pad_idx=2):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1] 
            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx)
            self.tgt_y = tgt[:, 1:] # used for loss calculation
            self.ntokens = (self.tgt_y != pad_idx).data.sum() # used for loss calculation

    @staticmethod
    def make_std_mask(tgt, pad_idx):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


class TrainState:
    """
    Track number of steps, examples, and tokens processed.
    """
    step: int = 0 # Steps in the current epoch
    accum_step: int = 0 # Number of gradient accumulation steps
    samples: int = 0 # Total number of examples used
    tokens: int = 0 # Total number of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """
    Train a single epoch.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            # implement as gradient accumulation - often used when the available hardware 
            # cannot handle the desired batch size due to memory constraints.
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | "
                    + "Tokens per Sec: %7.1f | Learning Rate: %6.1e"
                )
                % ( i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        
        del loss, loss_node

    return total_loss / total_tokens, train_state
            

def learning_rate(step, model_size, factor, warmup):
    """
    We have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing using KL divergence.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Args:
            size: number of classes
            padding_idx: index of padding token
            smoothing: smoothing rate
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)