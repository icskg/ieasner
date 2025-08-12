from utils.common import init_weights

import os
import torch
from torch import nn
from transformers import AlbertModel, AutoModel, BertModel


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim % n_head == 0
        in_head_dim = dim // n_head
        out = dim if out is None else out
        assert out % n_head == 0
        out_head_dim = out // n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """

        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w


class ALBertIndustry(nn.Module):
    def __init__(self, bert_name: str, bert_dim: int, num_hiddens: int, num_classes: int, gamma: float = 1.0,
                 delta: float = 1.0):
        super(ALBertIndustry, self).__init__()
        self.bert_name = bert_name
        self.is_chinese = True if 'chinese' in bert_name else False
        self.bert_dim = bert_dim
        self.num_classes = num_classes
        self.sigmoid_gamma = gamma
        self.sigmoid_delta = delta
        model_path = os.path.join(os.path.dirname(__file__), '..', 'huggingface', bert_name)
        if "albert" in bert_name:
            self.bert = AlbertModel.from_pretrained(model_path)
        else:
            self.bert = BertModel.from_pretrained(model_path)
        self.attn = MultiHeadBiaffine(bert_dim, out=2 * num_hiddens)

        self.start_linear = nn.Sequential(
            nn.Linear(bert_dim, num_hiddens, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_hiddens, 1, bias=True)
        )
        self.end_linear = nn.Sequential(
            nn.Linear(bert_dim, num_hiddens, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_hiddens, 1, bias=True)
        )

        self.span_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_hiddens, num_classes, bias=True)
        )

        # self.cnn = nn.Conv2d(2 * num_hiddens, num_hiddens, kernel_size=3, padding=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(2 * num_hiddens, 4 * num_hiddens, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * num_hiddens, 2 * num_hiddens, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(2 * num_hiddens, num_hiddens, kernel_size=3, padding=1)
        )

        self.start_linear.apply(init_weights)
        self.end_linear.apply(init_weights)
        self.span_linear.apply(init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # with torch.no_grad():
        embeddings = self.bert(input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids).last_hidden_state

        span_matrix = self.attn(embeddings, embeddings)
        span_matrix = self.cnn(span_matrix).permute(0, 2, 3, 1)
        # span matrix shape (batch_size, seq_len, seq_len, embedding_dim)

        start_logits = torch.sigmoid(self.start_linear(embeddings)).squeeze(-1)
        end_logits = torch.sigmoid(self.end_linear(embeddings)).squeeze(-1)

        seq_len = span_matrix.shape[1]
        start_matrix = start_logits.unsqueeze(-1).unsqueeze(1).repeat(1, 1, 1, seq_len).permute(0, 2, 3, 1)
        end_matrix = end_logits.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1).permute(0, 2, 3, 1)

        start_matrix = 1. / (1 + torch.exp(-start_matrix + 0.5))
        end_matrix = 1. / (1 + torch.exp(-end_matrix + 0.5))

        span_matrix = span_matrix * start_matrix * end_matrix

        # span_logits shape (batch_size, num_classes, seq_len, seq_len)
        span_logits = torch.sigmoid(self.span_linear(span_matrix)).permute(0, 3, 1, 2)

        return start_logits, end_logits, span_logits
