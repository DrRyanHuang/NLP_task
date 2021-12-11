

from gensim.models import word2vec


if __name__ == "__main__":
    model = word2vec.Word2Vec.load(
        './embedding/word2vec_100features_10minwords_5win_30epoch.model')
    print(model)
    print(model.wv.similarity('happy', 'exciting'))
    print(model.wv.similarity('happy', 'sad'))
    print(model.wv.most_similar('exciting', topn=10))
    print(model.wv['happy'])
