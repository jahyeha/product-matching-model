#--------------------------#
# Sentence Embedding: LSTM
#--------------------------#

from main import Main

class SentenceEmbedding:

    def __init__(self, input_size, hidden_size, num_layers, bias=True, bidirectional=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.lstm = None
        self.main = Main('data.pickle')
        self.seq_words_vecs = self.main.seq_words_vecs
        self.tensors = self.padding()

    def build(self):
        import torch.nn as nn
        lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bidirectional)
        output, (hn, cn) = lstm(self.tensors)
        return output

    def padding(self):
        import torch
        tensors = torch.nn.utils.rnn.pad_sequence(self.seq_words_vecs, batch_first=True)
        return tensors

if __name__ == '__main__':
    SE = SentenceEmbedding(2, 10, 1) #(input dim, hidden, layer)
    result = SE.build()
    print(result[0])
