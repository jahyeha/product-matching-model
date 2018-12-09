import pandas as pd
import re

# Loading data #
def load_data():
    g_data = pd.read_csv('dataset/goods.csv', encoding='CP949', usecols=['g_modelno', 'g_modelnm'])
    p_data = pd.read_csv('dataset/pricelist.csv', encoding='CP949', usecols=['pl_no', 'pl_goodsnm', 'pl_modelno'])
    p_data = p_data[p_data['pl_modelno'] != 0]  # Not-matched goods names
    return (g_data, p_data)

def make_goodsnms_lst(): # total_goods_nms
    g_data, p_data = load_data()
    pl_goodsnms = list(p_data['pl_goodsnm'].values)
    g_goodsnms = list(g_data['g_modelnm'].values)
    total_goods_nms = pl_goodsnms + g_goodsnms
    return total_goods_nms

def create_basic_dict():
    g_data, p_data = load_data()
    g_modelnm, g_modelno = g_data['g_modelnm'], g_data['g_modelno']

    # modelno_to_goodsnm & modelno_to_goodsnms #
    # 1. modelno_to_goodsnm
    modelno_to_goodsnm = dict()
    # └> e.g. {modelno: 'modelno에 해당하는 상품명(1 goods name)', ..}
    for i in range(len(g_modelnm)):
        modelno_to_goodsnm[g_modelno[i]] = g_modelnm[i]

    # 2. modelno_to_goodsnms
    modelno_to_goodsnms = dict()
    # └>{modelno: [modelno에 매칭된 pl 상품명(1), pl 상품명(2), ..], ..}
    for _, row in p_data.iterrows():
        (key, val) = row['pl_modelno'], row['pl_goodsnm']
        # └ (key: 모델(카탈로그)번호, val: key에 매칭되는 plicelist 상품명 하나)
        if key not in modelno_to_goodsnms:
            modelno_to_goodsnms[key] = [val]
        else:
            modelno_to_goodsnms[key].append(val)
    return (modelno_to_goodsnm, modelno_to_goodsnms)

def tokenize_sentence(sentence):
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