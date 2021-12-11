import os
from util import read_pkl, write_pkl


class WordDict:
    
    def __init__(self, word_voc_path=None, dataset=None, **kwarg):
        '''
        @Para:
            word_voc_path  :  专指 `data/aclImdb/imdb.vocab`
            kwarg 中:
                max_time        :  词典中的最大出现次数 TODO: NotImplemented
                min_time        :  词典中的最小出现次数 TODO: NotImplemented
                max_words_count :  词典中的最多词语数量(默认8000), 低频词过多, 样本过于稀疏不太好
                load_old        :  是否直接加载旧的词典(默认 False), 若加载则直接跳过其他步骤
        '''
        #TODO: 通过给定的 dataset 自学习 word Dict
        
        load_old = kwarg.get("load_old", False)
        
        model_path = "model_save/wordDict_id2word.pkl"
        if load_old and os.path.exists(model_path):
            self.id2word = read_pkl(model_path)
            self.word2id = {v:k for k, v in self.id2word.items()}
            
            self.length = len(self.id2word)
            return
            
        
        if dataset is not None:
            raise NotImplementedError()
        
        assert os.path.exists(word_voc_path)
        with open(word_voc_path, "r") as f:
            all_word = f.read().split()
        
        # 加上没见过的词 和 PAD 填充词, 一个排 0, 一个排 1
        all_word = ["OOV", "PAD"] + all_word
        max_words_count = kwarg.get("max_words_count", 8000)
        all_word = all_word[:max_words_count+2]
        
        
        self.id2word = {i:word for i, word in enumerate(all_word)}
        self.word2id = {v:k for k, v in self.id2word.items()}
        
        write_pkl("wordDict_id2word.pkl", self.id2word)
        
        self.length = len(self.id2word)



    def seqence2vector(self, seq_list, max_len=150):
        # 将一个单词列表转化为字典
        
        res = [ self.word2id["PAD"] ] * max_len
        
        for i in range(max_len):
            
            if i==len(seq_list):
                break
            
            temp = self.word2id.get(seq_list[i], self.word2id["OOV"])
            res[i] = temp
            
        return res
            
    
        
    def vector2seqence(self, vector):
        
        res = []
        for id_ in vector:
            res.append(self.id2word[id_])
        return res

    
    def __call__(self, seq_list, max_len=150):
        
        return self.seqence2vector(seq_list, max_len)


    def __len__(self):
        return self.length






if __name__ == "__main__":
    x = WordDict("data/aclImdb/imdb.vocab", load_old=True)
    vec = x.seqence2vector(['the', 'soon', "parts"])
    seq = x.vector2seqence(vec)
    vec_ = x(['the', 'soon', "parts", "yuanyi"], 12)