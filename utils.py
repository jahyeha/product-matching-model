from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from numpy import array
import toyData
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

def clean_sentence(sentence):
    # input: 'sentence'
    #   │- lower, replace 특수문자, etc.
    # output: 'cleaned sentence'
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
    return sent

def tokenize_sentence(sentence): # claen&Tokenize
    # input: 'sentence'
    #   │- lower, replace 특수문자, etc.
    #   │- tokenize
    # output: ['token', 'token', 'token', ..]
    sent = clean_sentence(sentence)
    return sent.split()

#---------------------------------------------------#
### For Toy Data ###
# 1. Documents 단위
def make_docsNlabels(max_size):
    # output
    #   ├ docs (list) = ['PL 상품명', 'PL 상품명',..]
    #   │                   ↓ Matching
    #   └ labels (array) = [modelno, modelno]
    #toy_dict = ToyData().create_toy_dict(max_size=max_size)
    toy_dict = toyData.ToyData().create_toy_dict(max_size=max_size)
    cleaned_toyDict = dict()
    for modelno, pl_nms in toy_dict.items():
        cleaned_lst = list()
        for modelnm in pl_nms:
            cleaned = clean_sentence(modelnm)
            cleaned_lst.append(cleaned)
        cleaned_toyDict[modelno] = cleaned_lst
    # └> cleaned_toyDict => 불용어 제거된 toy_dict (형태는 동일!)

    docs, labels = list(), list() # labels=> array로 변환
    for modelno, pl_nms in cleaned_toyDict.items():
        for i in range(len(pl_nms)):
            docs.append(pl_nms[i])
            labels.append(modelno)
            # 같은 정답인 것들이 쭉 들어가게 됨 => shuffle 꼭 해주기!
    labels = array(labels)
    #print("# length of docs: {} | labels: {}".format(len(docs), len(labels)))
    return (docs, labels)

# ↓ ↓ ↓

def create_word_idx_dict(docs):
    # [in] docs (list) = ['PL 상품명', 'PL 상품명',..]
    # [out] word_idx_dict: {'단어': index #, '단어', index #, ..}
    #                   └> {'크림':1, '50ml':2, '아이오페':3, ...}
    # *size of vocabs => len(word_idx_dict)
    t = Tokenizer()
    t.fit_on_texts(docs)
    word_idx_dict = t.word_index
    return word_idx_dict

def make_encoded_lst(docs):
    # [in] docs (list) = ['PL 상품명', 'PL 상품명',..]
    #             └ e.g. ['정품 히말라야 크림', '정품 히말라야 보습', ..]
    # [out] encoded_lst (list of lists)
    #             └ e.g. [[1, 2, 3], [1, 2, 5],..]
    t = Tokenizer()
    encoded_lst = t.texts_to_sequences(docs)
    return encoded_lst
#          ----------↓
def make_padded_lst(encoded_lst):
    # [in] encoded_lst (e.g.) = [[1,2], [1,2,5,6]]
    #           ↓ Padding by max length of list (==4)
    # [out] padded_lst (e.g.) = [[1,2,0,0],
    #                            [1,2,5,6]]
    max_length = len(max(encoded_lst, key=len))
    padded_lst = pad_sequences(encoded_lst,
                               maxlen=max_length,
                               padding='post')
    return padded_lst

#---------------------------------------------------#
# 2. Sentence 단위
def sent_to_encoded(sentence):
    # in: '(정품) 히말라야 인텐시브 고수분크림 150ml 영양'
    # out: [13, 6, 11, 49, 9, 47]
    t = Tokenizer()
    tokenized_lst = tokenize_sentence(sentence)
    print(tokenized_lst)
    encoded_res = list()
    for token in tokenized_lst:
        (docs, labels) = make_docsNlabels(max_size=50) ##### 수정 필요!
        word_idx_dict = create_word_idx_dict(docs)
        encoded_res.append(word_idx_dict[token])
    return encoded_res


if __name__ == '__main__':
    test_sent = '(정품) 히말라야 인텐시브 고수분크림 150ml 영양'
    print(sent_to_encoded(test_sent))