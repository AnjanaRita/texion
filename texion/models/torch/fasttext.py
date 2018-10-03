import torch
import numpy as np
from torch import nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, label_size, embeddings=None,
                 requires_grad=True, linear_hidden_size=2000):
        super().__init__()
        self.linear_hidden_size = linear_hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.label_size = label_size
        self.embeddings = embeddings
        self.requires_grad = requires_grad
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        if embeddings is not None:
            print('load embedding...')
            self.encoder.weight = nn.Parameter(
                embeddings, requires_grad=requires_grad)

        self.content_fc = nn.Sequential(nn.Linear(embedding_dim, linear_hidden_size),
                                        nn.BatchNorm1d(linear_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(linear_hidden_size, label_size))

    def forward(self, content):
        content = torch.autograd.Variable(
            torch.arange(0, 2500).view(10, 250)).long()
        content_ = torch.mean(self.encoder(content), dim=1)
        out = self.content_fc(content_.view(content_.size(0), -1))
        return out
