from lstm import MyLSTM
from gensim.models import FastText
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import pickle
import utils

# Setting
max_size = 50
max_seq_len = 30

# Loading..
# ├ ⑴Toy dict.(pre-stored @toyData.py)
# ├ ⑵Word Embedding Model (pre-trained @wordEmbedding.py)
# └ ⑶LSTM model
with open('dictionary/toyDict.pickle', 'rb') as handle:
    toy_dict = pickle.load(handle)
fastText = FastText.load('model/FastText.bin')
lstm = load_model('model/lstm.h5')
#----------------------------------------------------#

myLSTM = MyLSTM(toy_dict=toy_dict, embedding_model=fastText)
(X_train, Y_train, X_val, Y_val, X_test, Y_test, toy_train_dict) = myLSTM.split_train_test()
# e.g. X_train[:3] => N차원 벡터 3개(N=300) e.g. [[300D vec], [300D vec], [300D vec]]
#      Y_train[:3] => [15, 8, 48]
X_train = sequence.pad_sequences(np.array(X_train), maxlen=max_seq_len)
index_dict = myLSTM.create_index_dict()

#-------------------run LSTM model-------------------#
print("✱ Run LSTM model...")
myLSTM.run_lstm(max_seq_len=max_seq_len) # run LSTM
#---------------------Prediction---------------------#
X_new = X_train[:50]
Y_new = Y_train[:50] # 정답
Y_hat = list(lstm.predict_classes(X_new))
print("\nY_hat: {},\nY_new: {}\n".format(Y_hat, Y_new))

for i in range(len(Y_hat)):
    pl_no = toy_train_dict[i]
    pl_nm = utils.PL_basic_dict()[pl_no]
    predicted = index_dict[Y_hat[i]][1]
    prediction_res = (Y_hat[i] == Y_new[i]) # 예측값 == 정답값
    if prediction_res == True:
        print("\t✔ | {} ===> {}".format(pl_nm, predicted))
    else:
        print("\t✖ | {} ===> {} | ✪정답: {}".format(pl_nm, predicted,index_dict[Y_new[i]][1]))
