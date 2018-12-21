from lstm import MyLSTM
from gensim.models import FastText
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import pickle
import utils

# Setting
max_size, max_seq_len = 50, 30
(modelno_to_goodsnm, modelno_to_goodsnms) = utils.model_basic_dict()

# Loading..
# ├ ⑴Toy dict.(pre-stored @toyData.py)
# ├ ⑵Word Embedding Model (pre-trained @wordEmbedding.py)
# └ ⑶LSTM model (pre-stored @lstm.py)
with open('dictionary/toyDict.pickle', 'rb') as handle:
    toy_dict = pickle.load(handle)
fastText = FastText.load('model/FastText.bin')
preLSTM = load_model('model/lstm.h5')

LSTM = MyLSTM(toy_dict=toy_dict, embedding_model=fastText)
(X_train, Y_train, X_val, Y_val, X_test, Y_test, toy_train_dict) = LSTM.split_train_test()
X_test = sequence.pad_sequences(np.array(X_test), maxlen=max_seq_len)
index_dict = LSTM.create_index_dict()

#---------------------Prediction---------------------#
X_new = X_test[:50]
Y_new = Y_test[:50]
Y_hat = list(preLSTM.predict_classes(X_new))
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
