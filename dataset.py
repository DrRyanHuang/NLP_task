import os
from torch.utils.data import Dataset, DataLoader


class IMDB(Dataset):

    def __init__(self, root, mode="train"):

        '''
        root 目录下组织格式:

            ├─test
            │  ├─neg
            │  └─pos
            └─train
                ├─neg
                ├─pos
                └─unsup
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


    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        with open(path) as f:
            text = f.read()
        return text, label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    # 调试一下看看对不
    dataset = IMDB(r"data/aclImdb")
    print(dataset[-1])