from utils import util
from model.SVM import SVMClassifer

if __name__ == "__main__":
    review_demo = "this moive is so excited. I love it."
    review_embedding = util.transform_review2average_embedding(review_demo)
    svm_classifier = SVMClassifer("rbf")
    res = svm_classifier.infer(review_embedding)
    print(res)
