#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:23:09 2017

@author: akli
"""

import os.path
import gensim,logging
import doc2vec
from sklearn.cluster import DBSCAN
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pathCorpus = "../Corpus"
path = "../word2vecModels"

dicoW2V = dict()
#Parcours des dossiers contenant les espaces w2v
for w2vdir in os.listdir(path):
    pathModel = path+"/"+w2vdir+"/w2v"
    pathPhraser = path+"/"+w2vdir+"/phrases"
    phraser = None
    #Verification presence du modele
    if os.path.isfile(pathModel) :
        w2v = gensim.models.Word2Vec.load(pathModel)
        #Verification presence phraser (cas des termes complexes)
        if(os.path.isfile(pathPhraser)):
            phraser = gensim.models.phrases.Phraser.load(pathPhraser)
        dicoW2V[w2vdir] = doc2vec.corpus2vec(pathCorpus,w2v,phraser)

dicoDBSCAN = dict()
dicoPred = dict()
for espace in dicoW2V.keys():
    dicoDBSCAN[espace] = DBSCAN()
    dicoPred[espace] = dicoDBSCAN[espace].fit_predict(dicoW2V[espace][0])
    print("%s : %d clusters"%(espace,len(np.unique(dicoPred[espace]))))

repartition = []
for espace in dicoPred.keys():
    repartition.append([np.sum(dicoPred[espace] == clusterIndex)for clusterIndex in np.unique(dicoPred[espace])])

purete = []

for espace in dicoPred.keys():
    for cluster in np.unique(dicoPred[espace]):
        annoncesCluster = np.where(dicoPred[espace] == cluster)
        repartitionClasses = np.bincount(dicoW2V[espace][1][annoncesCluster])
        classeMajoritaire = np.argmax(repartitionClasses)
        
        
    
