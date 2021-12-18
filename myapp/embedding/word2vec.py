import logging

from nltk import data
import data_prepare
# Initialize and train the model (this will take some time)
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# Set values for various parameters
num_features = 100    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 10       # Number of threads to run in parallel
window_size = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
sg = 1  # skip_gram
epoch = 30


if __name__ == '__main__':
    sentences = data_prepare.get_sentences()
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features,
                              min_count=min_word_count, window=window_size, sample=downsampling, sg=sg, epochs=epoch)

    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "myapp/embedding/word2vec_100features_10minwords_5win_30epoch.model"
    model.save(model_name)
