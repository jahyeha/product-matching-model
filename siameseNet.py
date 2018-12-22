from gensim.models import FastText
from numpy import zeros
import utils
from toyData import ToyData
import random
import numpy as np
from keras.preprocessing import sequence
from keras.backend import tensorflow_backend as K
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from time import time
import datetime
import matplotlib.pyplot as plt


class SiameseNet:
    def __init__(self, max_size):
        self.max_size = max_size
        self.toy_dict = ToyData().create_toy_dict(max_size)
        self.docs, self.labels = utils.make_docsNlabels(max_size)
        self.word_idx_dict = utils.create_word_idx_dict(self.docs)
        self.vocab_size = len(self.word_idx_dict)
        self.model_idx_dict = utils.model_idx_dict(max_size)
        self.embeddings_idx, self.embeddings_mat = self.make_embeddings_mat()

    def run_siamese_net(self):
        #---------------------------#
        maxlen = 43
        input_dimension = 300
        n_hidden = 50
        batch_size = 256
        n_epoch = 20
        gradient_clipping_norm = 1.25
        # ---------------------------#
        embeddings_mat = self.open_embeddings_mat()
        (train_left, train_right, train_label, val_left, val_right, val_label, test_left, test_right, test_label) = self.split_train_test()

        left_input = Input(shape=(maxlen,), dtype='int32')
        right_input = Input(shape=(maxlen,), dtype='int32')
        embedding_layer = Embedding(self.vocab_size,
                                    input_dimension,
                                    weights=[embeddings_mat],
                                    input_length=maxlen,
                                    trainable=False)
        # ↓ Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # ↓ Since this is a siamese network,
        #   both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        # Pack it all up into a model
        malstm = Model([left_input, right_input], [malstm_distance])
        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)

        malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # Start training
        training_start_time = time()
        malstm_trained = malstm.fit([train_left, train_right],
                                    train_label,
                                    batch_size=batch_size,
                                    nb_epoch=n_epoch,
                                    validation_data=([val_left, val_right], val_label))

        print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(
            seconds=time() - training_start_time)))

        plt.plot(malstm_trained.history['acc'])
        plt.plot(malstm_trained.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot loss
        plt.plot(malstm_trained.history['loss'])
        plt.plot(malstm_trained.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        scores = malstm.evaluate([test_left, test_right], test_label, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def exponent_neg_manhattan_distance(self, left, right):
        return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

    def split_train_test(self):
        left_lst, right_lst, labels_lst = self.split_left_right_label()
        padded_left_lst, padded_right_lst = self.encode_n_pad()
        train_end = int(len(padded_left_lst) * 0.6)
        val_end = int(len(padded_left_lst) * 0.8)

        train_left = np.array(padded_left_lst[:train_end], dtype=float)
        train_right = np.array(padded_right_lst[:train_end], dtype=float)
        train_label = np.array(labels_lst[:train_end], dtype=int)

        val_left = np.array(padded_left_lst[train_end:val_end], dtype=float)
        val_right = np.array(padded_right_lst[train_end:val_end], dtype=float)
        val_label = np.array(labels_lst[train_end:val_end], dtype=int)

        test_left = np.array(padded_left_lst[val_end:], dtype=float)
        test_right = np.array(padded_right_lst[val_end:], dtype=float)
        test_label = np.array(labels_lst[val_end:], dtype=int)

        return (train_left, train_right, train_label, val_left, val_right, val_label, test_left, test_right, test_label)

    def encode_n_pad(self):
        # encoding& padding
        encoded_left_lst = list()
        encoded_right_lst = list()
        (left_lst, right_lst, labels_lst) = self.split_left_right_label()
        for pl_goodsnm in left_lst:
            pl_goodsnm = utils.sent_to_encoded(pl_goodsnm, self.max_size)
            encoded_left_lst.append(pl_goodsnm)
        for model_nm in right_lst:
            model_nm = utils.sent_to_encoded(model_nm, self.max_size)
            encoded_right_lst.append(model_nm)

        # ↓ Padding
        padding_size = 43
        padded_left_lst = sequence.pad_sequences(encoded_left_lst, padding_size)
        padded_right_lst = sequence.pad_sequences(encoded_right_lst, padding_size)
        return (padded_left_lst, padded_right_lst)

    def split_left_right_label(self):
        arr_total_lst_labels = np.array(self.shuffle_total_lst_labels())
        left_lst = arr_total_lst_labels[:, 0]
        right_lst = arr_total_lst_labels[:, 1]
        labels_lst = arr_total_lst_labels[:, 2]
        return (left_lst, right_lst, labels_lst)

    def shuffle_total_lst_labels(self):
        matching_lst, matching_lst_label = self.create_matching_lst()
        not_matching_lst, not_matching_lst_label = self.create_not_matching_lst()
        total_lst_labels = matching_lst_label + not_matching_lst_label
        random.shuffle(total_lst_labels)
        return total_lst_labels

    def create_matching_lst(self):
        matching_lst = list()
        # ㄴ[ [pl_goodsnm, modelnm(==pl_goodsnm)], ..]
        idx = 0
        for modelno, pl_lst in self.toy_dict.items():
            for i in range(len(pl_lst)):
                matching_lst.append([pl_lst[i][1], self.model_idx_dict[idx][1]])
            idx += 1

        matching_lst_label = list()
        # ㄴ[ [pl_goodsnm, modelnm(==pl_goodsnm), 1], ..]
        for lst in matching_lst:
            lst += [1]
            matching_lst_label.append(lst)
        return (matching_lst, matching_lst_label)

    def create_not_matching_lst(self):
        not_matching_lst = list()
        # ㄴ[ [pl_goodsnm, modelnm(!=pl_goodsnm)], ..]
        not_matching_lst_label = list()
        # ㄴ[ [pl_goodsnm, modelnm(!=pl_goodsnm), 0], ..]
        model_idx_dict = utils.model_idx_dict(self.max_size)
        idx = 0
        for modelno, pl_lst in self.toy_dict.items():
            for tup in pl_lst:
                nums_not_me = list(range(0, self.max_size))
                nums_not_me.remove(idx)
                random_val = random.choice(nums_not_me)
                not_matching_lst.append([tup[1], model_idx_dict[random_val][1]])
            idx += 1

        for lst in not_matching_lst:
            lst += [0]
            not_matching_lst_label.append(lst)
        return (not_matching_lst, not_matching_lst_label)

    def open_embeddings_mat(self):
        return np.load('npy/embeddings_mat.npy') # (3195, 300)

    def save_embeddings_mat(self):
        np.save('npy/embeddings_mat', np.array(self.embeddings_mat))
        print("*****embeddings_mat.npy 저장 완료*****")

    def make_embeddings_mat(self):
        word_embed_vec_size = 300
        embeddings_idx = FastText.load('model/fastText.bin')
        embeddings_mat = zeros((self.vocab_size + 1, word_embed_vec_size))
        for word, i in self.word_idx_dict.items():
            embedding_vector = embeddings_idx[word]
            if embedding_vector is not None:
                embeddings_mat[i] = embedding_vector
        return (embeddings_idx, embeddings_mat)


if __name__ == '__main__':
    max_labels_num = 10
    #SiameseNet(max_labels_num).save_embeddings_mat()
    SiameseNet(max_labels_num).run_siamese_net()

