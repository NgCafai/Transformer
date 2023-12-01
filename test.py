import torch
from models.utils import subsequent_mask
from models.transformer import Transformer, make_model

def inference_test():
    vocab = ['<pad>', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    vocab_size = len(vocab)

    test_model = make_model(vocab_size, vocab_size, N=2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (batch_size, seq_len)
    src_mask = torch.ones(1, 1, 10) # (batch_size, 1, seq_len)

    memory = test_model.do_encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.do_decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src).fill_(next_word)], dim=1)

    print("Example Untrained Model Prediction: ", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    run_tests()

