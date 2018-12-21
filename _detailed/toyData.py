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
        # 1. modelno에 매칭이 가장 많이된 순서대로 N개 자르기
        #  > modelno_to_goodsnms의 value 길이를 기준으로 내림차순 정렬, 슬라이싱
        modelno_to_length = dict()
        # └> {modelno: length of a list, ..}
        for key, val in self.modelno_to_goodsnms.items():
            modelno_to_length[key] = len(val)

        sort_n_sliced = sorted(
            modelno_to_length.items(),
            key=operator.itemgetter(1),
            reverse=True)[:max_size]
        # └> e.g. [(12712082, 894), (11109083, 740), (1760331, 740),..] | length: 100 (=N)

        # 2. Creating Toy dict.
        toy_dict = dict()
        for i in range(len(sort_n_sliced)):
            modelno = sort_n_sliced[i][0]
            toy_dict[modelno] = self.modelno_to_goodsnms[modelno]
        # └> e.g. {modelno: [(pl_no, pl_goodsnm),(pl_no2, pl_goodsnms),..], ..}
        return toy_dict