from toyData import ToyData
from lstm import MyLSTM
from gensim.models import FastText
#------------------#
max_size = 50
max_seq_len = 30
#------------------#

toy_dict = ToyData().create_toy_dict(max_size=max_size)
fastText = FastText.load('model/FastText.bin')
lstm = MyLSTM(toy_dict=toy_dict, embedding_model=fastText)
# TEST
print(lstm.run_lstm(max_seq_len=max_seq_len))