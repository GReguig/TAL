#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:48:50 2017

@author: akli
"""
import numpy as np
from nltk.corpus import stopwords
from Generator import DocGen


def doc2vec(doc,model,stop_words):
    docVec = np.zeros((model.vector_size))
    cpt = 0.0
    for word in doc: 
        if (word not in stop_words) and (word in model.wv.vocab.keys()):
            docVec = docVec+model[word] 
            cpt+=1.0
    if(cpt == 0):
        cpt = 1
    return docVec/cpt

def corpus2vec(filepath,model,phraser=None,unique = False):
    languages = ['french', 'english', 'german', 'spanish']
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
            stop_words.append(w)        
    d = DocGen(filepath)
    labels = d.categories
    files = d.files
    if phraser:
        d = phraser[d]
    if not unique : 
        res =  np.asarray([doc2vec(doc,model,stop_words) for doc in d])
    else:
        res =  np.asarray([doc2vec(set(doc),model,stop_words) for doc in d])
    return res,np.asarray(labels),files
"""
corpusVec,labels = corpus2vec("../Corpus",w2v,unique=True)

phraser = gensim.models.phrases.Phraser.load("/home/akli/TAL/application/word2vecModels/bigramAnnonces/phrases")
gen = DocGen("../Corpus")
labels = gen.categories
d = phraser[gen]
"""