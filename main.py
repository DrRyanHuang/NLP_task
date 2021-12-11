import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 

from tovector import WordDict
from model.LSTM import LSTM
from dataset import IMDB, return_dataloader
from util_dl import log_train, log_test, model_save, model_load


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_dic = WordDict("data/aclImdb/imdb.vocab", load_old=True)
# dateset_tr = IMDB(r"data/aclImdb", "train", transform=word_dic)
train_loader = return_dataloader(r"data/aclImdb", "train", transform=word_dic, batch_size=64)
test_loader = return_dataloader(r"data/aclImdb", "test", transform=word_dic, batch_size=128)


model = LSTM(len(word_dic), 
             embedding_dim=300,
             padding_idx=word_dic.word2id['PAD'],
             bidirectional=True).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.02)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


def test(epoch, ep):
    
    loss = 0
    right = 0
    model.eval()
    with torch.no_grad():
        for idx, (_, data, label) in enumerate(test_loader): 
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = F.nll_loss(output, label).item()
            pred = torch.max(output, dim=-1)[1]
            right += torch.eq(label, pred).sum().item()
        
        loss = loss / len(test_loader.dataset)
        right = right / len(test_loader.dataset)
        
        log_str = log_test(epoch, ep, loss, 100*right)
        print(log_str)
    model.train()
            

def train(epoch=200, test_step=20, model_dir=None, save_step=50, load=True):
    
    if load:
        # 这样读取保存的模型
        ep = model_load(model, optimizer, "model_save/LSTM_50.pth")
    else:
        ep = 0
    
    while(ep<epoch):

        for idx, (_, data, label) in enumerate(train_loader):
            
            data = data.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        lr = scheduler.get_last_lr()[0]
        log_str = log_train(epoch, ep+1, loss.item(), lr)
        print(log_str)
        
        if not (ep+1)%test_step:
            test(epoch, ep+1)
            
        if not (ep+1)%save_step:
            model_save(model, optimizer, ep, model_dir)
            
        ep += 1
        
    model_save(model, optimizer, ep, model_dir)
            

train(epoch=100, model_dir="./model_save")