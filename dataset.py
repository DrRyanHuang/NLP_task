import os
import numpy as np
from util import tokenize
import torch
from torch.utils.data import Dataset, DataLoader


class _IMDB(Dataset):

    def __init__(self, root, mode="train", seed=1234):

        '''
        root 目录下组织格式:

            ├─test
            │  ├─neg
            │  └─pos
            └─train
                ├─neg
                ├─pos
                └─unsup
        seed : 用来 shuffle 的种子
        '''

        # 参数有效性检验
        assert mode in ["train", "test"]
        assert os.path.exists(root)
        data_path = os.path.join(root, mode)
        assert os.path.exists(data_path)

        # 读入数据
        pos_path = os.path.join(data_path, "pos")
        self.pos_data_list = [os.path.join(pos_path, x) for x in os.listdir(pos_path)]

        neg_path = os.path.join(data_path, "neg")
        self.neg_data_list = [os.path.join(neg_path, x) for x in os.listdir(neg_path)]

        self.data_list = [(x, 1) for x in self.pos_data_list] + \
                         [(x, 0) for x in self.neg_data_list]
        np.random.seed(seed)
        np.random.shuffle(self.data_list)


    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        with open(path, encoding="utf-8", ) as f:
            raw_text = f.read()
        word_list = tokenize(raw_text)
        return raw_text, word_list, label

    def __len__(self):
        return len(self.data_list)




class IMDB(Dataset):
    
    def __init__(self, root, mode="train", p=0.2, transform=None):
        
        assert 0<p<1
        
        # p : 测试集的比例
        dataset_train = _IMDB(root, "train")
        dataset_test = _IMDB(root, "test")
        self.dataset = dataset_test + dataset_train
        
        self.transform = transform
        
        length = len(self.dataset)
        
        if mode == "train":
            self.start = 0
            self.length = int(length*(1-p))
        elif mode == "test":
            self.start = int(length*(1-p))
            self.length = length - int(length*(1-p))
        else:
            print("------- 你考虑清楚再和我说话 -------")

    
    def __getitem__(self, idx):
        
        if(len(self) <= idx):
            raise IndexError("------- 索引越界, 检查IMDB数据类的使用情况 -------")
        if(idx < 0):
            idx = len(self) + idx
            if idx < 0:
                raise IndexError("------- 索引越界, 负数的绝对值太大" + \
                                 "检查IMDB数据类的使用情况 -------")
        raw_out = list(self.dataset[self.start + idx])
        if self.transform is not None:
            raw_out[1] = self.transform(raw_out[1])
        
        return raw_out
    
    def __len__(self):
        return self.length
    


def __collate_fn(batch):
    
    batch = list(zip(*batch))
    
    text_tensor = torch.LongTensor(batch[1])
    label = torch.LongTensor(batch[2])
    
    return batch[0], text_tensor, label



def return_dataloader(root, mode, transform=None, **kwarg):
    """
    @Param:
        kwarg:
            batch_size
            num_workers
            collate_fn
            drop_last

    """
    batch_size = kwarg.get("batch_size", 128)
    kwarg['batch_size'] = batch_size
    
    dataset = IMDB(root, mode, transform=transform)
    return DataLoader(dataset, 
                      collate_fn=__collate_fn, 
                      **kwarg)


if __name__ == "__main__":
    # 调试一下看看对不
    
    dateset_train = IMDB(r"data/aclImdb", "train")
    dateset_test = IMDB(r"data/aclImdb", "test")
    
    print(dateset_test[0])