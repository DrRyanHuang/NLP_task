import re
import os
import pickle
import torch
import datetime

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


def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
def log_train(EPOCH, ep, loss, lr):
    
    log_str = '[Train] [{}] [Epoch: {}/{}]\tLoss: {:.6f}\tLearning_rate:: {:.6f}'
    return log_str.format(current_time(),
                          ep, EPOCH, loss, lr)



def log_test(EPOCH, ep, loss, p):
    
    log_str = '[Test] [{}] [Train Epoch: {}/{}]: Avgloss: {:.4f}, Accuracy: {:.2f}%\n'
    return log_str.format(current_time(),
                          ep, EPOCH, loss, p)





def model_save(model, optim, epoch, save_path=None):
    if save_path is None:
        return False
    
    state = {'net':model.state_dict(), 
             'optimizer':optim.state_dict(), 
             'epoch':epoch}
    name = model.NAME + "_{}.pth".format(epoch)
    path = os.path.join(save_path, name)
    torch.save(state, path)
    return True


def model_load(model, optim, model_path):
    
    if not os.path.exists(model_path):
        print("[Load Model Error] : `{}` not exist".format(model_path))
        return 0
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    optim.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    
    return start_epoch