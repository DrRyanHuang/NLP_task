import torch
import torch.nn as nn
import torch.optim as optim

from tovector import WordDict
from model.LSTM import LSTM
from dataset import IMDB, return_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_dic = WordDict(load_old=True)
# dateset_tr = IMDB(r"data/aclImdb", "train", transform=word_dic)
train_loader = return_dataloader(r"data/aclImdb", "train", transform=word_dic)

for x, y, z in train_loader:
    
    print(len(x))
    
# data_batch = torch.LongTensor(data_batch)

model = LSTM(len(word_dic), 
             padding_idx=word_dic.word2id['PAD'],
             bidirectional=True)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# out = model(data_batch)