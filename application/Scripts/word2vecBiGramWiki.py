#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:58:56 2017

@author: akli
"""

import gensim, logging
from Generator import DataGen
import os.path

path = u"../Corpus"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
modelPath = u"../word2vecModels/bigramWiki/w2v"
phrasesPath = u"../word2vecModels/bigramWiki/phrases"

if(not os.path.isfile(modelPath)):
    d = DataGen(path)
    phrases = gensim.models.Phrases(d)
    bigram = gensim.models.phrases.Phraser(phrases)
    bigram.save(phrasesPath)
    w2v = gensim.models.Word2Vec.load(u"/home/akli/TAL/w2vWiki/frwiki.gensim") 
    w2v.model_trimmed_post_training = False
    a = list(bigram[d])
    w2v.build_vocab(a)
    w2v.train(a)
    w2v.save(modelPath)
else:
    w2v = gensim.models.Word2Vec.load(modelPath)
    print("Modèle existant chargé")