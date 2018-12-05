import pickle
# Loading data.pickle
with open('data.pickle', 'rb') as f:
    toy_dict = pickle.load(f)

(keys, values) = toy_dict.keys(), toy_dict.values()
print(len(keys), len(values))
