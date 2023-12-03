import math
import torch
import copy
from torch import nn
from models.attention import MultiHeadAttention
from models.utils import subsequent_mask

class Transformer(nn.Module):
    """
    The Transformer model: A standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Args:
            encoder: nn.Module, a stack of N EncoderLayer
            decoder: nn.Module, a stack of N DecoderLayer
            src_embed: nn.Sequential, composed of Embeddings and PositionalEncoding, for input sequence
            tgt_embed: nn.Sequential, composed of Embeddings and PositionalEncoding, for output sequence
            generator: nn.Module, used to predict the next token
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: (batch_size, seq_len_src)
            tgt: (batch_size, seq_len_tgt)
            src_mask: (batch_size, 1, seq_len_src)
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.do_decode(self.do_encode(src, src_mask), src_mask, tgt, tgt_mask)

    def do_encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def do_decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout_prob=0.1, max_len=5000):
    """
    Helper: Construct a model from hyperparameters.
    """
    func_copy = copy.deepcopy
    attention = MultiHeadAttention(h, d_model, dropout_prob)
    feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout_prob)
    positional_encoding = PositionalEncoding(d_model, dropout_prob, max_len)
    model = Transformer(
        encoder=Encoder(EncoderLayer(d_model, func_copy(attention), func_copy(feed_forward), dropout_prob), N),
        decoder=Decoder(DecoderLayer(d_model, func_copy(attention), func_copy(attention), func_copy(feed_forward), dropout_prob), N),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), func_copy(positional_encoding)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab_size), func_copy(positional_encoding)),
        generator=Generator(d_model, tgt_vocab_size)
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    
# Generator --------------------------------------------------------------------
class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab_size):
        """
        Args:
            vocab_size: size of the vocabulary, that is, total number of unique tokens
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return nn.functional.log_softmax(self.linear(x), dim=-1)


# Components for both encoder and decoder ---------------------------------------
class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SubLayer(nn.Module):
    """
    Do pre-layer normalization for input, and then run multi-head attention or feed forward,
    and finally do the residual connection.
    """
    def __init__(self, d_model, dropout_prob=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, main_logic):
        # main_logic: multi-head attention or feed forward
        x_norm = self.norm(x)
        return x + self.dropout(main_logic(x_norm))



# Encoder ----------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Core encoder is a stack of N EncoderLayer.
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.
    """
    def __init__(self, d_model, attention, feed_forward, dropout_prob):
        super().__init__()
        self.d_model = d_model
        self.attention = attention
        self.feed_forward = feed_forward
        self.sub_layers = nn.ModuleList([SubLayer(d_model, dropout_prob) for _ in range(2)])

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: 
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        x = self.sub_layers[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.sub_layers[1](x, self.feed_forward)
        return x


# Decoder ----------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Core decoder is a stack of N DecoderLayer.
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory is the output of the Encoder
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """
    def __init__(self, d_model, self_attention, src_attention, feed_forward, dropout_prob):
        super().__init__()
        self.d_model = d_model
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sub_layers = nn.ModuleList([SubLayer(d_model, dropout_prob) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            memory: (batch_size, seq_len, d_model)
            src_mask: 
            tgt_mask:
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # self-attention: query, key, value are all from x
        x = self.sub_layers[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # src-attention: query is from x, while key and value are from the output of the Encoder
        x = self.sub_layers[1](x, lambda x: self.src_attention(x, memory, memory, src_mask))
        x = self.sub_layers[2](x, self.feed_forward)
        return x
    

# Position-wise feed forward ---------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))


# Embedding --------------------------------------------------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        Args:
            vocab_size: size of the vocabulary, that is, total number of unique tokens
        """
        super().__init__()
        self.lookup_table = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lookup_table(x) * math.sqrt(self.d_model)
    

# Positional encoding -----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        # Compute the positional encodings once in log space.
        positional_encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # size: (max_len, 1)

        # Equivalent to 1 / (10000 ^ (2i / d_model)) in the paper
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        ) # size: (d_model / 2,)

        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        positional_encodings = positional_encodings.unsqueeze(0) # size: (1, max_len, d_model)
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        return self.dropout(x)
