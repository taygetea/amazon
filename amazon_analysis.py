import json
from gensim.models import Word2Vec
from vec2pca import multitokenize, train
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from collections import defaultdict
from multiprocessing import Process, Pool, cpu_count
import sys

logging.basicConfig(format='%asctime)s %(levelname)s: %message)s', level=logging.INFO, datefmt='%H:%M:%S')

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
        if counter%100000 == 0:
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
# tokenized_reviews = to_wordlist(rtext)
tokenized_reviews = multitokenize(rtext, processes=16)

print("train test split")
fullset = np.array(list(zip(tokenized_reviews, rratings)))
np.random.shuffle(fullset)
splitpos = int(len(fullset)*0.8)

training, test = fullset[:splitpos,:], fullset[splitpos:,]
for badnum in [a for a in range(len(training[:,0])) if not training[a,0]]:
    training[badnum,0] = [""]
for badnum in [a for a in range(len(test[:,0])) if not test[a,0]]:
    test[badnum,0] = [""]
print("training model")
model = Word2Vec.load("amazon_model")

prcomps = dict(zip(model.index2word, PCA().fit_transform(pd.DataFrame(model[model.index2word]))))

# got these functions off Kaggle. Averages vectors for each word in a set of words.

def makeFeatureVec(words, model, prcomps, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, prcomps[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model=model, prcomps=prcomps, num_features=100):
    import sys
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        #if counter%10000. == 0.:
        #   print("Review %d of %d" % (counter, len(reviews)))
        #   sys.stdout.flush()
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, prcomps, num_features)
        counter = counter + 1
    #print("Review %d of %d" % (counter, len(reviews)))
    #sys.stdout.flush()
    return reviewFeatureVecs

print("starting multiprocess averaging")
def mapred(func, data, processes=cpu_count()):
    def chunks(l, n):
        n = int(len(l)/n)
        for i in range(0, len(l), n):
            yield l[i:i+n]
    print("initializing pool")
    pool = Pool(processes=processes)
    print("partitioning input")
    partitioned = chunks(data, processes*100)
    print("averaging")
    mapped = pool.imap_unordered(func, partitioned)
    aggregator = []
    print("aggregating")
    for section in mapped:
        aggregator.extend(section)
    return aggregator



trainDataVecs = mapred(getAvgFeatureVecs, training[:,0])
# testDataVecs = mapred(getAvgFeatureVecs, test[:,0])
import pdb
pdb.set_trace()
np.save('trainDataVecs', trainDataVecs)
for idx, row in enumerate(trainDataVecs):
    if not np.isfinite(row).all():
        trainDataVecs[idx] = np.zeros_like(trainDataVecs[0])
for idx, row in enumerate(testDataVecs):
    if not np.isfinite(row).all():
        testDataVecs[idx] = np.zeros_like(testDataVecs[0])
np.save('trainDataVecs', trainDataVecs)
np.save('testDataVecs', testDataVecs)
np.save('test', test)
np.save('training', training)
print("fitting model...")
lm = LogisticRegressionCV(n_jobs=-1, max_iter=500,solver='sag')
lm_fit = lm.fit(trainDataVecs, list(training[:,1]))
lm_preds = lm.predict(testDataVecs)
print("score: ", lm_fit.score(testDataVecs, list(test[:,1])))
print("MSE: ", np.mean((lm_fit.predict(testDataVecs) - test[:,1] ** 2)))
df = pd.DataFrame(data={"text": list(map(" ".join, test[:,0])),
                   "actual": test[:,1],
                   "result": lm_preds}).sort_values("result")
print(df[['result', 'actual']].astype('int').corr())
# df.to_csv("amazon_logistic")

