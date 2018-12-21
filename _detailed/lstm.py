from utils import model_basic_dict
from utils import load_data
from utils import tokenize_sentence
from random import shuffle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import FastText
import pickle


class MyLSTM:

    def __init__(self, toy_dict, embedding_model):
        self.toyData = toy_dict
        self.embedding = embedding_model
        self.g_data, self.p_data = load_data()
        self.pl_label_lst, self.vec_label_lst = self.create_label_lst()
        self.modelno_to_goodsnm = model_basic_dict()[0]

    def run_lstm(self, max_seq_len):
        (X_train, Y_train, X_val, Y_val, X_test, Y_test, toy_train_dict) = self.split_train_test()
        X_train = sequence.pad_sequences(np.array(X_train), maxlen=max_seq_len)
        X_val = sequence.pad_sequences(np.array(X_val), maxlen=max_seq_len)
        X_test = sequence.pad_sequences(np.array(X_test), maxlen=max_seq_len)

        model = Sequential()
        model.add(LSTM(200, input_shape=(30, 300)))
        #model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train, epochs=3, batch_size=100, validation_data=(X_val, Y_val))

        # Plot accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        model.save('model/lstm.h5')
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def split_train_test(self):
        list_len = len(self.vec_label_lst)
        shuffle(self.vec_label_lst)
        idx_num_train, idx_num_val = int(list_len * 0.6), int(list_len * 0.8)
        # Ratio | 6:2:2 (train: val: test)
        toy_train = self.vec_label_lst[:idx_num_train] # ●●●●●●○○○○
        # └> e.g. [ [([[vector], [vector],...], pl_no), 0], .. ]
        # 18.12.10 UPDATE -----------------------------------------#
        toy_train_dict = dict()
        for i in range(len(toy_train)):
            toy_train_dict[i] = toy_train[i][0][1] #pl_no
        # └> {0: 1st pl_no, 1: 2nd pl_no, ...} => for PREDICTION @main.py
        #----------------------------------------------------------#
        toy_val = self.vec_label_lst[idx_num_train:idx_num_val] # ●●●●●●[●●]○○
        toy_test = self.vec_label_lst[idx_num_val:] # ●●●●●●●●[●●]
        print("--------------------------------------")
        print("train: {} | val: {} | test: {}".format(len(toy_train), len(toy_val), len(toy_test)))
        print("--------------------------------------")

        X_train, Y_train = list(), list()
        for lst in toy_train: #> toy_train
            X_train.append(lst[0][0]) # N-d vector set| a list of lists
            Y_train.append(lst[1]) # model index e.g. 0

        X_val, Y_val = list(), list()
        for lst in toy_val:
            X_val.append(lst[0][0])
            Y_val.append(lst[1])

        X_test, Y_test = list(), list()
        for lst in toy_test:
            X_test.append(lst[0][0])
            Y_test.append(lst[1])
        return (X_train, Y_train, X_val, Y_val, X_test, Y_test, toy_train_dict)

    def create_label_lst(self):
        # out: (pl_label_lst, vec_label_lst)
        pl_label_lst = list()
        # └> [[('히말라야 인텐시브 고수분크림 150ml  영양', pl_no), 0],..]
        num = 0
        for modelno, pl_nms in self.toyData.items():
            for i in range(len(pl_nms)):
                pl_label_lst.append([(pl_nms[i][1], pl_nms[i][0]), num])
            num += 1

        vec_label_lst = list()
        # ㄴ> [[([[vector], [vector],...], pl_no), 0], .. ] # Final
        #      -------------------------> 1 sentence
        for pl_label_set in pl_label_lst:
            # └> [('히말라야 인텐시브 고수분크림 150ml 영양', pl_no), 0]
            goodsnm = pl_label_set[0][0]
            # └> '히말라야 인텐시브 고수분크림 150ml 영양'
            tokenized = tokenize_sentence(goodsnm)
            # └> ['히말라야', '인텐시브', '고수분크림', '150ml', '영양']
            # └> [[vector], [vector], [vector], ...]
            for i in range(len(tokenized)):
                word_vec = self.embedding[tokenized[i]]
                tokenized[i] = word_vec
            vec_label_lst.append([(tokenized, pl_label_set[0][1]), pl_label_set[1]])
            #                ㄴ> [([tokenized], pl_no), label]
        return (pl_label_lst, vec_label_lst)

    def create_index_dict(self):
        # {0: (modelno, modelnm), 1: (modelno, modelnm), ... } = idx_dict
        idx_dict = dict()
        for i in range(len(self.toyData)):
            modelno = list(self.toyData.keys())[i]
            idx_dict[i] = (modelno, self.modelno_to_goodsnm[modelno])
        return idx_dict


if __name__ == '__main__':
    max_seq_len = 30
    with open('dictionary/toyDict.pickle', 'rb') as handle:
        toy_dict = pickle.load(handle)
    fastText = FastText.load('model/FastText.bin')
    # -------------------run LSTM model-------------------#
    print("✱ Run LSTM model...")
    lstm = MyLSTM(toy_dict=toy_dict, embedding_model=fastText)
    lstm.run_lstm(max_seq_len=max_seq_len) # run& save my LSTM model


# reference| https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/