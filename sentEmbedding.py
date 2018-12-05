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
        self.main = Main('data.pickle')
        self.seq_words_vecs = self.main.seq_words_vecs
        self.tensors = self.padding()

    def similarity(self, u, v):
        # Cosine Similarity
        import numpy as np
        import math
        u = np.asarray(list(u))
        v = np.asarray(list(v))
        return np.dot(u,v)/(math.sqrt(np.dot(u,u))*math.sqrt(np.dot(v,v)))

    def lstm(self):
        import torch.nn as nn
        model = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bidirectional)
        output, (hn, cn) = model(self.tensors)
        return output

    def padding(self):
        import torch
        tensors = torch.nn.utils.rnn.pad_sequence(self.seq_words_vecs, batch_first=True)
        return tensors



if __name__ == '__main__':
    SE = SentenceEmbedding(2, 10, 1) #(input dim, hidden, layer)
    result = SE.lstm()
    print(result[0])
    print(result[0][-1])
