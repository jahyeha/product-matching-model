import pandas as pd

# Loading data #
def load_data():
    g_data = pd.read_csv('dataset/goods.csv', encoding='CP949', usecols=['g_modelno', 'g_modelnm'])
    p_data = pd.read_csv('dataset/pricelist.csv', encoding='CP949', usecols=['pl_no', 'pl_goodsnm', 'pl_modelno'])
    p_data = p_data[p_data['pl_modelno'] != 0]  # Not-matched goods names
    return (g_data, p_data)
