from django.shortcuts import render
from django.shortcuts import HttpResponse

from myapp.model import LR
from myapp.model import SVM
from myapp.model import util
from myapp.model import statisic_model
from myapp.model import Bayes
# Create your views here.
SentiDict_Classifer = statisic_model.SentiDict()
LR_Classifier = LR.LRClassifer()
SVM_Classifier = SVM.SVMClassifer()
NB_Classifier = Bayes.NBClassifer()


def trans2word(count):
    if count == 0:
        return "neg"
    else:
        return "pos"


def model_res():
    res = {"SVM": "pos", "NB": "pos", "LR": "pos", "XLNet": "pos"}
    return res


def models(review, review_embedding):
    SentiDict_res = SentiDict_Classifer.infer(review)
    LR_res = LR_Classifier.infer(review_embedding)[0]
    SVM_res = SVM_Classifier.infer(review_embedding)[0]
    NB_res = NB_Classifier.infer(review_embedding)[0]
    res = {"SentiDict": trans2word(SentiDict_res), "LR": trans2word(LR_res), "SVM": trans2word(SVM_res),
           "NB": trans2word(NB_res), "LSTM": "111", "XLNet": "111"}
    return res


def models_res_with_prob(review, review_embedding):
    SentiDict_res = SentiDict_Classifer.infer(review)
    LR_res = LR_Classifier.infer_with_proba(review_embedding)[0]
    SVM_res = SVM_Classifier.infer_with_proba(review_embedding)[0]
    NB_res = NB_Classifier.infer_with_proba(review_embedding)[0]
    res = {"SentiDict": {"class": trans2word(SentiDict_res), "prob": "---"},
           "LR": {"class": trans2word(LR_res.argmax()), "prob": round(LR_res.max(), 5)},
           "SVM": {"class": trans2word(SVM_res.argmax()), "prob": round(SVM_res.max(), 5)},
           "NB": {"class": trans2word(NB_res.argmax()), "prob": round(NB_res.max(), 5)},
           "LSTM": {"class": "111", "prob": "---"},
           "XLNet": {"class": "111", "prob": "---"}}
    return res


def index(request):
    res = model_res()
    return render(request, "index.html", {'res': res})


def query(request):
    review = request.POST.get('user_input')
    # 数据处理
    review_embedding = util.transform_review2average_embedding(review)
    # infer
    # res = models(review, review_embedding)
    res = models_res_with_prob(review, review_embedding)
    return render(request, "query.html", {'review': review, 'res': res})
