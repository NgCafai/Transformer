"""
A simple copy task: Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.
"""
import torch
from torch.optim.lr_scheduler import LambdaLR
from train import *
from models.transformer import make_model
from models.utils import *

def data_gen(vocab, batch_size, n_batches, max_len=8):
    """
    Generate random data for a src-tgt copy task.
    """
    for i in range(n_batches):
        start_symbol_idx = vocab['<start>']
        end_symbol_idx = vocab['<end>']
        # Create a tensor with random integers between (start_symbol_idx + 1) and (end_symbol_idx - 1), size: (batch_size, 10)
        data = torch.randint(start_symbol_idx + 1, end_symbol_idx, size=(batch_size, max_len))

        data[:, 0] = start_symbol_idx
        data[:, -1] = end_symbol_idx
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, pad_idx=vocab['<pad>'])


class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """
    def __init__(self, generator, criterion):
        """
        Args:
            generator: nn.Module, used to generate the probability distribution over the target vocabulary

        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return loss.data * norm, loss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Predicts a translation using greedy decoding for simplicity.
    """
    memory = model.do_encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.do_decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def example_simple_model():
    # Example copy task for symbols: 'a' to 'k'
    vocab = {
        '<pad>': 0,  # Padding token
        '<start>': 1,  # Start of sequence token
        'a': 2,
        'b': 3,
        'c': 4,
        'd': 5,
        'e': 6,
        'f': 7,
        'g': 8,
        'h': 9,
        'i': 10,
        'j': 11,
        'k': 12,
        '<end>': 13  # End of sequence token
    }
    vocab_size = len(vocab)  # Should be 10 in this case

    criterion = LabelSmoothing(size=vocab_size, padding_idx=vocab['<pad>'], smoothing=0.0)
    model = make_model(vocab_size, vocab_size, N=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(vocab, batch_size=batch_size, n_batches=20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )

        model.eval()

        run_epoch(
            data_gen(vocab, batch_size=batch_size, n_batches=5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
    
    model.eval()
    max_len = 8
    src_sentence = ['<start>', 'a', 'b', 'c', 'i', 'j', 'k', '<end>']
    src = torch.LongTensor([[vocab[word] for word in src_sentence]])
    src_mask = torch.ones(1, 1, max_len)
    prediction = greedy_decode(model, src, src_mask, max_len, start_symbol=vocab['<start>'])

    idx_to_word = {idx: word for word, idx in vocab.items()}
    prediction_sentence = [idx_to_word[idx.item()] for idx in prediction[0]]
    print("Example Trained Model Prediction: ", prediction_sentence)


if __name__ == "__main__":
    example_simple_model()