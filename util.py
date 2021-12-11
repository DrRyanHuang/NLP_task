import re
import os
import pickle

# 分词的API
def tokenize(text):
    
    # 该函数取自:
    #    https://blog.csdn.net/Delusional/article/details/113357449
    
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]




def read_pkl(pickle_file):
    # 读取保存的 pkl 文件
    
    with open(pickle_file, 'rb') as f:
        data_dic = pickle.load(f)
    return data_dic




def write_pkl(pickle_file, data):
    # 将某些数据写成 pkl 
    if os.path.exists("./model_save"):
        pass
    else:
        os.mkdir("./model_save")
    pickle_file = os.path.join("./model_save", pickle_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)