import numpy as np
import gensim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report

import joblib
import util
import os


def labelize_reviews(reviews):
    for i, v in enumerate(reviews):
        # 预处理，每个评论最多只取100个词
        yield gensim.utils.simple_preprocess(v, max_len=100)


def build_word_vector(model, text, size=100):
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

# models


def NaiveBayes_classifer(x_train, y_train, x_test, y_test):
    file_name = "model_params/NaiveBayes_classifier.pkl"
    if os.path.exists(file_name):
        classifier = joblib.load(file_name)
    else:
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, file_name)
    print("=============================================")
    print("================NaiveBayes===================")
    y_pred = classifier.predict(x_test)
    y_pred = np.rint(y_pred)
    print(classification_report(y_test, y_pred, digits=4))
    print("=============================================\n")


def SVM_rbf_kernerl_classifer(x_train, y_train, x_test, y_test):
    file_name = "model_params/SVM_rbf_kernerl_classifier.pkl"
    if os.path.exists(file_name):
        classifier = joblib.load(file_name)
    else:
        classifier = SVC(probability=True)
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, file_name)
    print("=============================================")
    print("================SVM rbf===================")
    y_pred = classifier.predict(x_test)
    y_pred = np.rint(y_pred)
    print(classification_report(y_test, y_pred, digits=4))
    print("=============================================\n")


def SVM_linear_kernerl_classifer(x_train, y_train, x_test, y_test):
    file_name = "model_params/SVM_linear_kernerl_classifier.pkl"
    if os.path.exists(file_name):
        classifier = joblib.load(file_name)
    else:
        classifier = SVC(kernel="linear", probability=True)
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, file_name)
    print("=============================================")
    print("================SVM linear===================")
    y_pred = classifier.predict(x_test)
    y_pred = np.rint(y_pred)
    print(classification_report(y_test, y_pred, digits=4))
    print("=============================================\n")


def LR_classifer(x_train, y_train, x_test, y_test):
    # 逻辑回归
    file_name = "model_params/LR_classifier.pkl"
    if os.path.exists(file_name):
        classifier = joblib.load(file_name)
    else:
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, file_name)

    print("=============================================")
    print("================LogisticRegression===================")
    y_pred = classifier.predict(x_test)
    y_pred = np.rint(y_pred)
    print(classification_report(y_test, y_pred, digits=4))
    print("=============================================\n")


pos_reviews, neg_reviews = util.get_all_data_v2()
# 数据划分, 1代表积极情绪，0代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
x_train, x_test, y_train, y_test = train_test_split(
    np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)


x_train_tag = list(labelize_reviews(x_train))
x_test_tag = list(labelize_reviews(x_test))


model = word2vec.Word2Vec.load(
    'myapp/embedding/word2vec_100features_10minwords_5win_30epoch.model')

train_vecs = np.concatenate([build_word_vector(model,
                                               gensim.utils.simple_preprocess(z, max_len=100)) for z in x_train])
test_vecs = np.concatenate([build_word_vector(model,
                                              gensim.utils.simple_preprocess(z, max_len=100)) for z in x_test])
# train_vecs = scale(train_vecs)
# test_vecs = scale(test_vecs)


# SVM_linear_kernerl_classifer(train_vecs, y_train, test_vecs, y_test)
SVM_rbf_kernerl_classifer(train_vecs, y_train, test_vecs, y_test)
# LR_classifer(train_vecs, y_train, test_vecs, y_test)
# NaiveBayes_classifer(train_vecs, y_train, test_vecs, y_test)
