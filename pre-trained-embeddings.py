# This script is an example adapted from
# https://github.com/kavgan/nlp-in-practice/blob/master/pre-trained-embeddings/Pre-trained%20embeddings.ipynb

import warnings
warnings.filterwarnings('ignore')

import gensim.downloader as api

# download the model and return as object ready for use
model_glove_twitter = api.load("glove-twitter-25")

model_glove_twitter.wv.most_similar("pelosi",topn=10)

#what doesn't fit?
model_glove_twitter.wv.doesnt_match(["trump","bernie","obama","pelosi","orange"])

# show weight vector for trump and obama
model_glove_twitter["trump"],model_glove_twitter['obama']
#again, download and load the model
model_gigaword = api.load("glove-wiki-gigaword-100")

# find similarity
model_gigaword.wv.most_similar(positive=['dirty','grimy'],topn=10)
model_gigaword.wv.most_similar(positive=["summer","winter"],topn=10)




# LOAD a dataset and train
from gensim.models.word2vec import Word2Vec

# this loads the text8 dataset
corpus = api.load('text8')

# train a Word2Vec model
model_text8 = Word2Vec(corpus,iter=10,size=150, window=10, min_count=2, workers=10)  # train a model from the corpus

# similarity
model_text8.wv.most_similar("shocked")

# similarity between two different words
model_text8.wv.similarity(w1="dirty",w2="smelly")

# Which one is the odd one out in this list?
model_text8.wv.doesnt_match(["cat","dog","france"])
