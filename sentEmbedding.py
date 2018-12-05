class SentenceEmbedding:
    def __init__(self, input_size, hidden_size, num_layers, input, bias=True, bidirectional=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.input = input
        self.bidirectional = bidirectional
        self.lstm = None

    def build(self):
        import torch.nn as nn
        lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bidirectional)
        output, (hn, cn) = lstm(self.input)
        return output