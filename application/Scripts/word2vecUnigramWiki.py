#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:05:34 2017

@author: akli
"""

import gensim, logging
import os.path
from doc2vec import corpus2vec
from sklearn.cluster import DBSCAN
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
modelPath = u"../word2vecModels/unigramWiki/w2v"

if(not os.path.isfile(modelPath)):
    w2v = gensim.models.Word2Vec.load(u"/home/akli/TAL/w2vWiki/frwiki.gensim") 
    w2v.save(modelPath)
else:
    w2v = gensim.models.Word2Vec.load(modelPath)
    print("Modèle existant chargé")
    
corpus, labels,files = corpus2vec("../Corpus",w2v,unique=True)
eps = np.linspace(.01,.1,10)
repartitions = []
purete = []
for e in eps : 
    db = DBSCAN(eps=e,metric='cosine',algorithm='brute')
    pred = db.fit_predict(corpus)
    p = []
    for cluster in np.unique(db.labels_):
        categoriesCluster = labels[np.where(pred == cluster)]
        compte = np.bincount(categoriesCluster)
        p.append(np.max(compte)*1.0/np.sum(compte))
    purete.append(p)
    print("eps = %f ; %d clusters"%(e,len(np.unique(db.labels_))))
    repartitions.append(np.bincount(pred+1))
        