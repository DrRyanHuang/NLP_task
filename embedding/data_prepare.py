import os 
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
# nltk.download()
# global var
tokenizer=nltk.data.load('tokenizer/tokenizers/punkt/english.pickle')


# 取所有的数据做语料库，包括train and test中的labeled and unlabeled. 
# 读入数据
def get_data():
    data_root_path="data/aclImdb/aclImdb"
    labeled_data_list=[]
    unlabeled_data_list=[]
    # labeled data
    for dataset_path in ['train','test']:
        for type in ['pos','neg']:
            path=os.path.join(data_root_path,dataset_path+"/"+type)
            for x in os.listdir(path):
                with open(os.path.join(path,x)) as f:
                    text = f.read()
                    labeled_data_list.append(text)
    # unlabeled data
    unlabeled_path=os.path.join(data_root_path,"train/unsup")
    for x in os.listdir(unlabeled_path):
        with open(os.path.join(unlabeled_path,x)) as f:
            text = f.read()
            unlabeled_data_list.append(text)
    
    print("label data size:",len(labeled_data_list))
    print("unlabel data size:",len(unlabeled_data_list))
     
    return labeled_data_list,unlabeled_data_list



# 数据预处理
def transform_review2words(review, remove_stopwords=False):
    plain_review=BeautifulSoup(review).get_text()
    plain_review=re.sub("[^a-zA-Z]"," ",plain_review)
    words=plain_review.lower().split()
    if remove_stopwords:
        stops=set(stopwords.words('english'))
        words=[x for x in words if x not in stops]
    
    return words

# 将review转换为多个句子
def transform_review2sentences(review,tokenizer,remove_stopwords=False):
    raw_sentences=tokenizer.tokenize(review.strip())
    processed_sentences=[]
    for raw_sent in raw_sentences:
        processed_sentences.append(transform_review2words(raw_sent,remove_stopwords))
    return processed_sentences

def get_sentences():
    labeled_data,unlabeled_data=get_data()
    sentences=[]
    for review in labeled_data:
        sentences+=transform_review2sentences(review,tokenizer)
    for review in unlabeled_data:
        sentences+=transform_review2sentences(review,tokenizer)
    return sentences

if __name__== '__main__':
    get_sentences()
    # print(len(sentences))
    # print(sentences[0])
    # print(sentences[1])

