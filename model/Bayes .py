
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
import util


class NBClassifer():

    def __init__(self):
        self.model_save_path = "model_params/NaiveBayes_classifier.pkl"  # 模型保存的位置
        self.model = None

    def load(self):
        self.model = joblib.load(self.model_save_path)

    def train(self, x_train, y_train):
        self.model = GaussianNB()
        self.model.fit(x_train, y_train)
        joblib.dump(self.model, self.model_save_path)

    def eval(self, x_test, y_test):
        if self.model == None:
            if os.path.exists(self.model_save_path):
                self.model = joblib.load(self.model_save_path)
            else:
                self.train(x_test, y_test)  # 合理调用不会走到这里
        y_pred = self.model.predict(x_test)
        y_pred = np.rint(y_pred)
        print(classification_report(y_test, y_pred, digits=4))

    def infer(self, average_embedding):
        if self.model == None:
            self.load()
        y_pred = self.model.predict(average_embedding)
        return y_pred

    def infer_with_proba(self, average_embedding):
        if self.model == None:
            self.load()
        y_pred_with_proba = self.model.predict_proba(average_embedding)
        return y_pred_with_proba


if __name__ == "__main__":
    review_demo = "For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan 'The Skipper' Hale jr. as a police Sgt."
    # review_demo = "fuck fuck fuck"
    review_embedding = util.transform_review2average_embedding(review_demo)
    nb_classifier = NBClassifer()
    res = nb_classifier.infer(review_embedding)
    print(res)
