# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:46:11 2017

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

#### WORD2VEC####
modelPath = u"../word2vecModels/unigramAnnonces/w2v" 

if(not os.path.isfile(modelPath)):
    w2v = gensim.models.Word2Vec(d,sg=1,size=100,workers=4)
    w2v.save(modelPath)
    w2v.syn1neg
else : 
    w2v = gensim.models.Word2Vec.load(modelPath)
    print("Modèle existant chargé")

corpus, labels, files = corpus2vec("../Corpus",w2v,unique=True)
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
