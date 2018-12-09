from utils import load_data
from random import shuffle
import operator

# 1. Loading data #
print("===================START===================")
(g_data, p_data) = load_data()
print("○ length of p_data: {} | g_data: {}".format(len(p_data), len(g_data)))
total_goods_nms = list(p_data['pl_goodsnm'].values) + list(g_data['g_modelnm'].values)
g_modelnm, g_modelno = g_data['g_modelnm'], g_data['g_modelno']


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

#------------------#
max_size = 50
#------------------#
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

#==========================================================================================#
# UPDATE | 18.12.08. sat
# 3. Create dict. for Toy set
modelno_to_goodsnm = dict()
for i in range(len(g_modelnm)):
    modelno_to_goodsnm[g_modelno[i]] = g_modelnm[i]

toy_dict = dict() # N개
for i in range(len(sort_n_sliced)):
    modelno = sort_n_sliced[i][0]
    toy_dict[modelno] = modelno_to_goodsnms[modelno]
# └> e.g. toy_dict = {12712082:['정품 히말라야 인텐시브 고수분크림', '히말라야 인텐시브 고수분크림',..],..}

def toss_toyDict(): # => classification.py
    return toy_dict
def toss_modelno_to_goodsnm():
    return modelno_to_goodsnm

############### For Siame Net. ###############
# 4. Create a input list for TOY SET
# └> toy_input_lst = [[model_nm, pl_nm], [model_nam, pl_nm],...]
toy_input_lst = list()
for key in (toy_dict.keys()):
    for model_nm in toy_dict[key]:
        toy_input_lst.append([model_nm, modelno_to_goodsnm[key]])
#print("\n* length of toy_input_lst:", len(toy_input_lst))
#print("* toy_input_lst[:5]:", toy_input_lst[:5])


# 5. Split Train/Test Set (80:20)
list_len = len(toy_input_lst)
shuffle(toy_input_lst)
idx_num = int(list_len * 0.8)
toy_train, toy_test = toy_input_lst[:idx_num], toy_input_lst[idx_num:]
# └> (length) train: 18307 | test: 4577