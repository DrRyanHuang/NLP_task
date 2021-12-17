
# 分别加载情感词词典、否定词词典、程度副词词典
# 计算分数
# 计算准确率
# infer函数

# 如何自己训练一个情感词典呢？

from sklearn.metrics import classification_report
import gensim
import numpy as np
import util


class SentiDict():
    def __init__(self) -> None:
        self.senti_dict = dict()
        self.non_dict = dict()
        self.adv_dict = dict()
        self.init_dicts()

    def init_dicts(self):
        senti_path = "data/sentiment_dictionary/sentiment_dict_en_1.txt"
        # senti_path = "data/sentiment_dictionary/DictionaryIMDB.csv"
        adv_path = 'data/sentiment_dictionary/adv.txt'
        non_path = "data/sentiment_dictionary/non.txt"
        with open(senti_path) as f:
            for line in f.readlines():
                word = line.strip().split(',')[0]
                weight = eval(line.strip().split(',')[1])
                weight = round(weight, 4)
                self.senti_dict[word] = weight

        with open(adv_path) as f:
            for line in f.readlines():
                word = line.strip().split(',')[0]
                weight = eval(line.strip().split(',')[1])
                self.adv_dict[word] = weight

        with open(non_path) as f:
            for line in f.readlines():
                word = line.strip()
                weight = -1
                self.non_dict[word] = weight

    def eval_without_data(self):
        # 这里用IMDB的数据集做测试
        pos_reviews, neg_reviews = util.get_all_data_v2()
        # 数据划分, 1代表积极情绪，0代表消极情绪
        ys = np.concatenate(
            (np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
        xs = np.concatenate((pos_reviews, neg_reviews))
        y_pred = []
        for x in xs:
            y_pred.append(self.infer(x))
        print(classification_report(ys, y_pred, digits=4))

    def infer(self, review):
        review = gensim.utils.simple_preprocess(review, max_len=150)
        score = 0
        for i, word in enumerate(review):
            temp_score = 1
            if word in self.senti_dict.keys():
                temp_score *= self.senti_dict[word]
                i -= 1
                if i >= 0 and review[i] in self.adv_dict.keys():
                    temp_score *= self.adv_dict[review[i]]
                while i >= 0:
                    if review[i] in self.non_dict.keys():
                        temp_score *= -1
                        i -= 1
                    else:
                        break
                # sum
                score += temp_score
        # print(score)
        if score >= 0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    SDModel = SentiDict()
    # example1 = "The moive is good, i love it very much."
    # SDModel.infer(example1)
    SDModel.eval_without_data()
