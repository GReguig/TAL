#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:31:31 2017

@author: akli
"""

import gensim,logging
from Generator import DataGen
import os.path
from sklearn.cluster import DBSCAN
from doc2vec import corpus2vec
import numpy as np

path = u"../Corpus"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

d = DataGen(path)
modelPath = u"../word2vecModels/trigramAnnonces/w2v" 
phrasesPath = u"../word2vecModels/trigramAnnonces/phrases"

if(not os.path.isfile(modelPath)):
    phrases = gensim.models.Phrases(d)
    bigram = gensim.models.phrases.Phraser(phrases)
    trigram = gensim.models.Phrases(bigram[d])
    trigram.save(phrasesPath)
    w2v = gensim.models.Word2Vec(list(trigram[d]),sg=1,size=100,workers=4)
    w2v.save(modelPath)
else : 
    w2v = gensim.models.Word2Vec.load(modelPath)
    phraser = gensim.models.Phrases.load(phrasesPath)
    print("Modèle existant chargé")
    
    
corpus, labels,files = corpus2vec("../Corpus",w2v,phraser,unique=True)
eps = np.linspace(.01,.1,10)
repartitions = []
purete = []
for e in eps : 
    db = DBSCAN(eps=e,metric="cosine",algorithm='brute')
    pred = db.fit_predict(corpus)
    p = []
    for cluster in np.unique(db.labels_):
        categoriesCluster = labels[np.where(pred == cluster)]
        compte = np.bincount(categoriesCluster)
        p.append(np.max(compte)*1.0/np.sum(compte))
    purete.append(p)
    print("eps = %f ; %d clusters"%(e,len(np.unique(db.labels_))))
    repartitions.append(np.bincount(pred+1))