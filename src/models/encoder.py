import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.rnn import LayerNormLSTM


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):  # 位置编码
    def __init__(
        self, dropout, dim, max_len=5000
    ):  # TODO(sujinhua): save some data and make a test on it
        pe = torch.zeros(max_len, dim)  # 最大长度*维度的0矩阵 dim == hidden_size(bert) 512
        position = torch.arange(0, max_len).unsqueeze(1)  # 返回maxlen*1维序列张量
        div_term = torch.exp(
            (
                torch.arange(0, dim, 2, dtype=torch.float)
                * -(math.log(10000.0) / dim)  # 0到维数步长为2的一维序列张量
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TimeEncoding(nn.Module):  # 时间编码（版本1）
    """
    我感觉时间数据并不适合在embedding的时候进行处理，因为position是对输入embedding的一句话进行编码，而且计算的是各字词的相对位置，
    但是时间数据一句话只有一个对应的时间点，并不适合要采取类似的处理，更适合把时间数据单独弄成一块，用一层单独处理，也就是版本二的方法
    所以我觉得这个版本可以弃了= =
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)  # 最大长度*维度的0矩阵
        position = torch.arange(0, max_len).unsqueeze(1)  # 返回maxlen*1维序列张量
        div_term = torch.exp(
            (
                torch.arange(0, dim, 2, dtype=torch.float)
                * -(math.log(10000.0) / dim)  # 0到维数步长为2的一维序列张量
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class T2V(nn.Module):  # 时间编码(版本二) 这个版本应该是自定义了一个层，在最后构建模型的时候才加入的，语雀文档里还有一个tensorflow版
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W", shape=(1, self.output_dim), initializer="uniform", trainable=True
        )
        self.P = self.add_weight(
            name="P", shape=(1, self.output_dim), initializer="uniform", trainable=True
        )
        self.w = self.add_weight(
            name="w", shape=(1, 1), initializer="uniform", trainable=True
        )
        self.p = self.add_weight(
            name="p", shape=(1, 1), initializer="uniform", trainable=True
        )
        super(T2V, self).build(input_shape)

    def call(self, x):

        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)

        return K.concatenate([sin_trans, original], -1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb  # TODO(sujinhua): + date_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](
                i, x, x, ~mask
            )  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class RNNEncoder(nn.Module):
    def __init__(self, bidirectional, num_layers, input_size, hidden_size, dropout=0.0):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        print("before transpose", x.size())
        x = torch.transpose(x, 1, 0)
        print("after transpose", x.size())
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores
