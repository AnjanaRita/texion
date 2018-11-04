import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super().__init__()

        """
		Parameters:
		__________

		batch_size : int
            Batch size
		
        output_size : int 
            output size

		hidden_sie : int
            Size of the hidden_state of the LSTM
		
        vocab_size : int
            vocabulary size 
		
        embedding_length : int 
            Embedding dimension of GloVe word embeddings
		
        weights : str
            Pre-trained GloVe word_embeddings, 
		
		"""

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(
            weights, requires_grad=False)
        self.dropout = 0.8
        self.lstm = nn.LSTM(embedding_length, hidden_size,
                            dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2*hidden_size+embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        """ 
        Parameters:
        __________

        input_sentence: tuple 
            input sentence with shape (batch_size, num_sequences)

        batch_size : int,(default: None)
            Used for prediction on a single sentence after training (batch_size = 1)

        Returns:
        _______

        Output of the linear layer with shape = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentence)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(
                2, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(
                2, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0))

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)
        y = y.permute(0, 2, 1)
        y = F.max_pool1d(y, y.size()[2])
        y = y.squeeze(2)
        _output = self.label(y)
        return _output
