import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import FastText
import pandas as pd
import re
from utils import tokenize_sentence

#--------------------------#
# Word Embedding: FastText #
#--------------------------#

class WordEmbedding:

    def __init__(self, goods_nms_list):
        self.nameList = goods_nms_list
        self.tokensListSet = self.tokens_list_set()

    def model(self, minCnt, size, window):
        # size = N dim. vector
        model = FastText(
                self.tokensListSet,
                min_count=minCnt,
                size=size,
                window=window)
        model.save('model/fastText.bin')
        print(model)

    def tokens_list_set(self):
        # INPUT for FastText
        tokenized_goods_list = []
        for sent in self.nameList:
            # ㄴsent == '잎스네이처 마린콜라겐 50 워터젤 크림', 'D티엔 수분 퐁당 크림',..
            tokenized = tokenize_sentence(sent)
            # ㄴtokenized == ['잎스네이처', '마린콜라겐', '50', '워터젤', '크림'], ..
            tokenized_goods_list.append(tokenized)
            # ㄴtokenized_goods_list == [[tokenized], [],..]
        return tokenized_goods_list


if __name__ == '__main__':
    # p_data: pricelist.csv | using 3 columns (pl_no, pl_goodsnm, pl_modelno)
    # g_data: goods.csv | using 2 columns (g_modelno, g_modelnm)
    g_data = pd.read_csv('dataset/goods.csv', encoding='CP949', usecols=['g_modelno', 'g_modelnm'])
    p_data = pd.read_csv('dataset/pricelist.csv', encoding='CP949', usecols=['pl_no', 'pl_goodsnm', 'pl_modelno'])
    #p_data = p_data[p_data['pl_modelno'] != 0]
    print("* length of p_data: {} | g_data: {}".format(len(p_data), len(g_data)))

    total_goods_nms = list(p_data['pl_goodsnm'].values) + list(g_data['g_modelnm'].values)
    WE = WordEmbedding(total_goods_nms)
    myModel = WE.model(1, 300, 3) # SAVED @model/fastText.bin # => MemoryError