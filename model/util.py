import numpy as np
import os
import re
from bs4 import BeautifulSoup
import gensim
from gensim.models import word2vec
# 为机器学习算法喂数据
WORD2VEC_MODEL = word2vec.Word2Vec.load(
    './embedding/word2vec_100features_10minwords_5win_30epoch.model')


def get_all_data():
    # 有标签的数据
    data_root_path = "data/aclImdb/aclImdb"
    pos_reviews = []
    neg_reviews = []
    for dataset_path in ['train', 'test']:
        for type in ['pos', 'neg']:
            path = os.path.join(data_root_path, dataset_path+"/"+type)
            for x in os.listdir(path):
                with open(os.path.join(path, x)) as f:
                    text = f.read()
                    if type == 'pos':
                        pos_reviews.append(text)
                    else:
                        neg_reviews.append(text)

    # 数据划分, 1代表积极情绪，0代表消极情绪
    # x = np.concatenate((pos_reviews,neg_reviews))
    # y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    return pos_reviews, neg_reviews


def get_all_data_v2():
    # 有标签的数据
    data_root_path = "data/aclImdb/aclImdb_summary"
    pos_reviews = []
    neg_reviews = []

    for type in ['pos.txt', 'neg.txt']:
        path = os.path.join(data_root_path, type)
        with open(path, 'r') as f:
            if type == 'pos.txt':
                for line in f.readlines():
                    if len(line) != 0:
                        pos_reviews.append(line)
            else:
                for line in f.readlines():
                    if len(line) != 0:
                        neg_reviews.append(line)

    return pos_reviews, neg_reviews


def clean_data(text):
    plain_review = BeautifulSoup(text).get_text()
    plain_review = re.sub("[^a-zA-Z]", " ", plain_review)
    words = plain_review.lower()
    return words


def labelize_reviews(reviews):
    for i, v in enumerate(reviews):
        # 预处理，每个评论最多只取100个词
        yield gensim.utils.simple_preprocess(v, max_len=100)


def labelize_review(review):
    return gensim.utils.simple_preprocess(review, max_len=100)


def build_word_vector(text, model=WORD2VEC_MODEL, size=100):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model.wv[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def transform_review2average_embedding(review):
    review = labelize_review(review)
    return build_word_vector(review)


if __name__ == '__main__':
    # pos, neg = get_all_data()
    # print(len(pos))
    # print(len(neg))
    # # save
    # with open('data/aclImdb/aclImdb_summary/pos.txt', 'w') as f:
    #     for line in pos:
    #         f.write(line+"\n")
    # with open('data/aclImdb/aclImdb_summary/neg.txt', 'w') as f:
    #     for line in neg:
    #         f.write(line+"\n")
    # print("saved")
    p, n = get_all_data_v2()
    print(len(p))
