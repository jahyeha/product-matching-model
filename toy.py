import pandas as pd
import operator
import pprint

# 1. Loading data #
g_data = pd.read_csv('dataset/goods.csv', encoding='CP949', usecols=['g_modelno', 'g_modelnm'])
p_data = pd.read_csv('dataset/pricelist.csv', encoding='CP949', usecols=['pl_no', 'pl_goodsnm', 'pl_modelno'])
p_data = p_data[p_data['pl_modelno'] != 0] # Not-matched goods names
print("* length of p_data: {} | g_data: {}".format(len(p_data), len(g_data)))
total_goods_nms = list(p_data['pl_goodsnm'].values) + list(g_data['g_modelnm'].values)


# 2. Creating dictionaries #
modelno_to_goodsnms = dict()
# {modelno: [modelno에 매칭된 상품명1, 상품명2, ..], ..}
# └> pricelist를 돌면서 key를 "modelno", value를 "modelno에 매칭된 상품명들의 리스트"로 갖는 dict 생성

for _, row in p_data.iterrows():
    (key, val) = row['pl_modelno'], row['pl_goodsnm']
    # └ (key: 모델(카탈로그)번호, val: key에 매칭되는 plicelist 상품명 하나)
    if key not in modelno_to_goodsnms:
        modelno_to_goodsnms[key] = [val]
    else:
        modelno_to_goodsnms[key].append(val)

# dict. "modelno_to_goodsnms" => value의 길이를 기준으로 내림차순 Sorting & Slicing (toy set)
# └> modelno에 가장 매칭이 많이된 순서대로 N개 자르기
modelno_to_length = dict()
# └> {modelno: length of a list, ..}
for key, val in modelno_to_goodsnms.items():
    modelno_to_length[key] = len(val)
max_size = 50
sort_n_sliced = sorted(
                modelno_to_length.items(),
                key=operator.itemgetter(1),
                reverse=True
                )[:max_size]
# └> e.g. [(12712082, 894), (11109083, 740), (1760331, 740),..] | length: 100 (=N)

toy_dict = dict() # N개
for i in range(len(sort_n_sliced)):
    modelno = sort_n_sliced[i][0]
    toy_dict[modelno] = modelno_to_goodsnms[modelno]
# └> e.g. {12712082:['정품 히말라야 인텐시브 고수분크림', '히말라야 인텐시브 고수분크림',..],..}

# {modelno: [[sentence 1's tokens], [sentence 2's tokens],..] | mutually MATCHING
# └> e.g.{ 12712082: [['정품', '히말라야', '인텐시브', '고수분크림'],
#                     ['히말라야', '인텐시브', '고수분크림'], ...], ...}
from wordEmbedding import WordEmbedding
WE = WordEmbedding(total_goods_nms)
modelno_to_tokens_list_set = dict()
for key, val in toy_dict.items():
    tokens_list = list()
    for sent in val:
        tokens_list.append(WE.tokenize_sentence(sent))
    modelno_to_tokens_list_set[key] = tokens_list

# {modelno: [[[vector], [vector], ..],
#            [[vector], [vector], ..],..], ...} | vector: N-d word embedding vector
from gensim.models import FastText
myModel = FastText.load('model/model.bin') # 저장된 모델 불러오기
modelno_to_vecs = dict()
error_list = list()

for key, val in modelno_to_tokens_list_set.items():
    vec_lst_set = list()
    for lst in val:
        for i in range(len(lst)):
            try:
                lst[i] = myModel[lst[i]] #word embedding vector
            except KeyError:
                error_list.append(lst[i])
                continue
        vec_lst_set.append(lst)
    modelno_to_vecs[key] = vec_lst_set

print(len(error_list))


# 3. Convert dict to pickle #
# Save
import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(modelno_to_vecs, f, pickle.HIGHEST_PROTOCOL)
