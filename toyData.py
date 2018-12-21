import utils
import operator
import pickle

class ToyData():

    def __init__(self):
        self.g_data, self.p_data = utils.load_data()
        self.total_goods_nms = utils.make_goodsnms_lst()
        self.g_modelnm = self.g_data['g_modelnm']
        self.g_modelno = self.g_data['g_modelno']
        self.modelno_to_goodsnms = utils.model_basic_dict()[1]

    def save_toy_dict(self, max_size):
        toy_dict = self.create_toy_dict(max_size)
        with open('dictionary/toyDict.pickle', 'wb') as handel:
            pickle.dump(toy_dict, handel, protocol=pickle.HIGHEST_PROTOCOL)

    def create_toy_dict(self, max_size):
        modelno_to_length = dict()
        # └> {modelno: length of a list, ..}
        for key, val in self.modelno_to_goodsnms.items():
            modelno_to_length[key] = len(val)

        sort_n_sliced = sorted(
            modelno_to_length.items(),
            key=operator.itemgetter(1),
            reverse=True)[:max_size]

        toy_dict = dict()
        for i in range(len(sort_n_sliced)):
            modelno = sort_n_sliced[i][0]
            toy_dict[modelno] = self.modelno_to_goodsnms[modelno]
        return toy_dict


if __name__ == '__main__':
    max_catalog_size = 50
    ToyData().save_toy_dict(max_size=max_catalog_size)