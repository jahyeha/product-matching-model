import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#--------------------------#
# Word Embedding: FastText
#--------------------------#

class WordEmbedding:

    def __init__(self, goods_nms_list):
        self.nameList = goods_nms_list
        self.tokensListSet = self.tokens_list_set()

    def model(self, minCnt, size, window):
        # size = N dim. vector
        from gensim.models import FastText
        model = FastText(
            self.tokensListSet,
            min_count=minCnt,
            size=size,
            window=window)
        model.save('model/model.bin')
        print(model)

    def tokens_list_set(self):
        # INPUT for FastText
        tokenized_goods_list = []
        for sent in self.nameList:
            # ㄴsent == '잎스네이처 마린콜라겐 50 워터젤 크림', 'D티엔 수분 퐁당 크림',..
            tokenized = self.tokenize_sentence(sent)
            # ㄴtokenized == ['잎스네이처', '마린콜라겐', '50', '워터젤', '크림'], ..
            tokenized_goods_list.append(tokenized)
            # ㄴtokenized_goods_list == [[tokenized], [],..]
        return tokenized_goods_list

    def tokenize_sentence(self, sentence):
        import re
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

if __name__ == '__main__':
    # p_data: pricelist.csv | using 3 columns (pl_no, pl_goodsnm, pl_modelno)
    # g_data: goods.csv | using 2 columns (g_modelno, g_modelnm)
    g_data = pd.read_csv('dataset/goods.csv', encoding='CP949', usecols=['g_modelno', 'g_modelnm'])
    p_data = pd.read_csv('dataset/pricelist.csv', encoding='CP949', usecols=['pl_no', 'pl_goodsnm', 'pl_modelno'])
    #p_data = p_data[p_data['pl_modelno'] != 0]
    print("* length of p_data: {} | g_data: {}".format(len(p_data), len(g_data)))

    total_goods_nms = list(p_data['pl_goodsnm'].values) + list(g_data['g_modelnm'].values)
    WE = WordEmbedding(total_goods_nms)
    myModel = WE.model(1, 2, 3) # SAVED @model/model.bin