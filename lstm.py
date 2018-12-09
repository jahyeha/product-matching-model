from toyData import toss_toyDict
from toyData import modelno_to_goodsnm
from utils import load_data
from gensim.models import FastText
from random import shuffle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import re

class MyLSTM:

    def __init__(self):
        self.toyData = toss_toyDict()
        self.fastText = FastText.load('model/FastText.bin')
        self.g_data, self.p_data = load_data()
        self.pl_label_lst, self.vec_label_lst = self.create_label_lst()

    def run_lstm(self, max_seq_len):
        (X_train, Y_train, X_val, Y_val, X_test, Y_test) = self.split_train_test_set()
        X_train = sequence.pad_sequences(np.array(X_train), maxlen=max_seq_len)
        X_val = sequence.pad_sequences(np.array(X_val), maxlen=max_seq_len)
        X_test = sequence.pad_sequences(np.array(X_test), maxlen=max_seq_len)

        model = Sequential()
        model.add(LSTM(200, input_shape=(30, 300)))
        model.add(Dense(50, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_data=(X_val, Y_val))
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def split_train_test_set(self):
        total_goods_nms = list(self.p_data['pl_goodsnm'].values) + list(self.g_data['g_modelnm'].values)
        g_modelnm, g_modelno = self.g_data['g_modelnm'], self.g_data['g_modelno']
        list_len = len(self.vec_label_lst)
        shuffle(self.vec_label_lst)
        idx_num_train, idx_num_val = int(list_len * 0.6), int(list_len * 0.8)
        toy_train = self.vec_label_lst[:idx_num_train]
        toy_val = self.vec_label_lst[idx_num_train:idx_num_val]
        toy_test = self.vec_label_lst[idx_num_val:]
        print("train:", len(toy_train), "val:", len(toy_val), "test:", len(toy_test))

        X_train, Y_train = list(), list()
        for i in toy_train:
            X_train.append(i[0])
            Y_train.append(i[1])
        X_val, Y_val = list(), list()
        for i in toy_val:
            X_val.append(i[0])
            Y_val.append(i[1])
        X_test, Y_test = list(), list()
        for i in toy_test:
            X_test.append(i[0])
            Y_test.append(i[1])
        return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

    def create_label_lst(self):
        # out: (pl_label_lst, vec_label_lst)
        pl_label_lst = list()
        # └> [['(정품) 히말라야 인텐시브 고수분크림 150ml  영양', 0], ['(정품) 히말라야 인텐시브 고수분크림 150ml  영양크림/보습/스킨/인텐시브/수분크림', 0],..]
        num = 0
        for modelno, pl_nms in self.toyData.items():
            # modelno: 12712082
            # plnms: ['정품 히말라야 인텐시브 고수분크림', '히말라야 인텐시브 고수분크림',..]
            for i in range(len(pl_nms)):
                pl_label_lst.append([pl_nms[i], num])
            num += 1

        vec_label_lst = list()
        # └> [ [[vector], [vector],...], 0], ...] # Final
        #      -----------------------> 1 sentence
        for pl_label_set in pl_label_lst:
            # └> ['(정품) 히말라야 인텐시브 고수분크림 150ml 영양', 0]
            goodsnm = pl_label_set[0]
            # └> '(정품) 히말라야 인텐시브 고수분크림 150ml 영양'
            tokenized = self.tokenize_sentence(goodsnm)
            # └> ['(정품)', '히말라야', '인텐시브', '고수분크림', '150ml', '영양']
            # └> [[vector], [vector], [vector], ...]
            for i in range(len(tokenized)):
                word_vec = self.fastText[tokenized[i]]
                tokenized[i] = word_vec
            vec_label_lst.append([tokenized, pl_label_set[1]])
        return (pl_label_lst, vec_label_lst)

    def create_index_dict(self):
        # {0: (modelno, modelnm), 1: (modelno, modelnm), ... } = idx_dict
        idx_dict = dict()
        for i in range(len(self.toyData)):
            modelno = list(self.toyData.keys())[i]
            idx_dict[i] = (modelno, modelno_to_goodsnm[modelno])
        return idx_dict

    def tokenize_sentence(self, sentence):
        # input: 'sentence'
        #   │- lower, replace 특수문자, etc.
        #   │- tokenize
        # output: ['token', 'token', 'token', ..]
        sent = re.sub('[^0-9a-z가-힣]+', ' ', sentence.lower())
        p1 = re.compile(r'(?P<num>[0-9]+)\s+(?P<eng>[a-zA-Z]+)')
        sent = p1.sub('\g<num>\g<eng>', sent)
        p2 = re.compile(r'(?P<char>[가-힣a-zA-Z]+)(?P<quantity>[0-9]+)')
        sent = p2.sub('\g<char> \g<quantity>', sent)
        p3 = re.compile(r'(?P<gml>[가-힣a-zA-Z]+)x(?P<num>[0-9]+)')
        sent = p3.sub('\g<gml> \g<num>', sent)
        p4 = re.compile(r'(?P<kor>[가-힣]+)(?P<eng>[a-zA-Z]+)')
        sent = p4.sub('\g<kor> \g<eng>', sent)
        p5 = re.compile(r'(?P<eng>[a-zA-Z]+)(?P<kor>[가-힣]+)')
        sent = p5.sub('\g<eng> \g<kor>', sent)
        return sent.split()

if __name__ == "__main__":
    lstm = MyLSTM()
    lstm.run_lstm(max_seq_len=30)

# reference
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
