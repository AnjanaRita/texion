import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoonKimCnn(nn.Module):
    """
    sentence classification model  based on https://arxiv.org/abs/1408.5882 by Yoon Kim.
    """

    def __init__(self, vocabulary_size, embedding_size,
                 channel_out, filter_heights, dropout=0.5):
        """
        parameters: 
        __________

        vocabulary_size: int
            size of the vocabulary being used

        embedding_size: int
            size of the embeddings to be used.

        channel_out: int
            number of output channels

        filter_heights: tuple
            the heights of the convolution filters to be used (ex: (2, 3, 4)) 

        dropout: float
            dropout size

        returns: 
        _______

        YoonKimCnn torch model

        """
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.conv = nn.ModuleList(
            [nn.Conv2d(1, channel_out, (fh, embedding_size)) for fh in filter_heights])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channel_out*len(filter_heights), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        prob = F.sigmoid(x)
        return prob
