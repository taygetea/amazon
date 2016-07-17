import json
import logging
import pdb
import sys
from collections import defaultdict
from multiprocessing import Pool, Process, cpu_count

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity
from vec2pca import multitokenize, train

logging.basicConfig(format='%asctime)s %(levelname)s: %message)s',
                    level=logging.INFO, datefmt='%H:%M:%S')

infile = "reviews_Books_5_short.json"
if len(sys.argv) > 1:
    infile = sys.argv[1]
print("loading in file")
counter = 0
asins = defaultdict(lambda: 0)
rtext = []
rratings = []
with open(infile) as f:
    for review in f:
        counter += 1
        if counter % 100000 == 0:
            print("processed %d reviews" % counter)
        reviewjson = json.loads(review)
        if asins[reviewjson['asin']] < 30:
            rtext.append(reviewjson['reviewText'])
            rratings.append(reviewjson['overall'])
        asins[reviewjson['asin']] = asins[reviewjson['asin']] + 1
    print("processed %d reviews, done." % counter)


def to_wordlist(corpus):
    wordlists = []
    for document in corpus:
        wordlist = document
        for splitchar in '.,;:/()&!?-_+"':
            wordlist = ' '.join(wordlist.split(splitchar))
        wordlist = [word for word in wordlist.split(" ") if word]
        wordlists.append(wordlist)
    return wordlists
print("building wordlist")

tokenized_reviews = multitokenize(rtext, processes=16)

print("train test split")
fullset = np.array(list(zip(tokenized_reviews, rratings)))
np.random.shuffle(fullset)
splitpos = int(len(fullset) * 0.8)

training, test = fullset[:splitpos, :], fullset[splitpos:, ]
for badnum in [a for a in range(len(training[:, 0])) if not training[a, 0]]:
    training[badnum, 0] = [""]
for badnum in [a for a in range(len(test[:, 0])) if not test[a, 0]]:
    test[badnum, 0] = [""]
print("training model")
model = Word2Vec.load("amazon_model")

prcomps = dict(zip(model.index2word, PCA().fit_transform(
    pd.DataFrame(model[model.index2word]))))

# adapted from
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
def make_feature_vec(words, model, prcomps, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, prcomps[word])
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

# adapted from
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
def get_average_feature_vecs(reviews, model=model, prcomps=prcomps, num_features=100):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(
            review, model, prcomps, num_features)
        counter += 1
    return review_feature_vecs

print("starting multiprocess averaging")

def mapred(func, data, processes=cpu_count()):
    def chunks(l, n):
        n = int(len(l) / n)
        for i in range(0, len(l), n):
            yield l[i:i + n]
    print("initializing pool")
    pool = Pool(processes=processes)
    print("partitioning input")
    partitioned = chunks(data, processes * 100)
    print("averaging")
    mapped = pool.imap_unordered(func, partitioned)
    aggregator = []
    print("aggregating")
    for section in mapped:
        aggregator.extend(section)
    return aggregator


train_data_vecs = mapred(get_average_feature_vecs, training[:, 0])

np.save('train_data_vecs', train_data_vecs)
for idx, row in enumerate(train_data_vecs):
    if not np.isfinite(row).all():
        train_data_vecs[idx] = np.zeros_like(train_data_vecs[0])
for idx, row in enumerate(test_data_vecs):
    if not np.isfinite(row).all():
        test_data_vecs[idx] = np.zeros_like(test_data_vecs[0])
np.save('train_data_vecs', train_data_vecs)
np.save('test_data_vecs', test_data_vecs)
np.save('test', test)
np.save('training', training)
print("fitting model...")
lm = LogisticRegressionCV(n_jobs=-1, max_iter=500, solver='sag')
lm_fit = lm.fit(train_data_vecs, list(training[:, 1]))
lm_preds = lm.predict(test_data_vecs)
print("score: ", lm_fit.score(test_data_vecs, list(test[:, 1])))
print("MSE: ", np.mean((lm_fit.predict(test_data_vecs) - test[:, 1] ** 2)))
df = pd.DataFrame(data={"text": list(map(" ".join, test[:, 0])),
                        "actual": test[:, 1],
                        "result": lm_preds}).sort_values("result")
print(df[['result', 'actual']].astype('int').corr())
