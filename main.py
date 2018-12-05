import pickle
import torch

class Main:
    def __init__(self, file_name):
        self.fn = file_name
        self.dict = self.open_pickle()
        self.seq_words_vecs = self.create_seqvecs_lst(self.dict)

    def open_pickle(self):
        with open(self.fn, 'rb') as f:
            dict = pickle.load(f)
            # └> {modelno: [ [[N-d word vec], [word vec]],...], modelno: [],.., ..}
            #                ----------------------------> 1 sentence
            return dict

    def create_seqvecs_lst(self, dict):
        seq_words_vecs = list()
        for keys, vals in dict.items():
            for val in vals:
                seq_words_vecs.append(torch.tensor(val))
        return seq_words_vecs


main = Main('data.pickle')
#print(main.seq_words_vecs[0])