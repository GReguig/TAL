#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:05:23 2017

@author: akli
"""

import gensim,logging
from Generator import DataGen
import os
import sys
from doc2vec import corpus2vec,doc2vec
from nltk.corpus import stopwords
import shutil
import numpy as np
import codecs
from sklearn.metrics.pairwise import cosine_similarity

def main(arg="../cv.txt"):
    if not arg : 
        print("Aucun fichier spécifié")
        return
    cvpath = str(arg)
    print("Ouverture du fichier %s"%(cvpath))
    f = codecs.open(cvpath,encoding="utf-8")
    t = f.read()
    f.close()
    t = t.lower().translate(str.maketrans('', '', '.!\"#?$%&(")*,-:;<=>@[]^_`{|}~*')).replace("\'"," ").split()
    print("Fichier correctement lu")
    print("Chargement de l'espace de représentation")
    path = u"../Corpus"
    modelPath = u"../word2vecModels/trigramAnnonces/w2v" 
    phrasesPath = u"../word2vecModels/trigramAnnonces/phrases"
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    d = DataGen(path)
    
    if(not os.path.isfile(modelPath)):
        phrases = gensim.models.Phrases(d)
        bigram = gensim.models.phrases.Phraser(phrases)
        trigram = gensim.models.Phrases(bigram[d])
        trigram.save(phrasesPath)
        w2v = gensim.models.Word2Vec(list(trigram[d]),sg=1,size=100,workers=4)
        w2v.save(modelPath)
    else : 
        w2v = gensim.models.Word2Vec.load(modelPath)
        
        #phraser = gensim.models.phrases.Phraser.load(phrasesPath)
        phraser = None 
        print("Modèle existant chargé")
    languages = ['french', 'english', 'german', 'spanish']
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
            stop_words.append(w)
    print("Lecture des annonces d'emploi")
    corpus, labels,files = corpus2vec("../Corpus",w2v,phraser)
    files = np.array(files)
    print("Sélection des annonces les plus intéressantes")
    distances = 1 - cosine_similarity(doc2vec(t,w2v,stop_words),corpus)
    args = np.argsort(distances)[0]
    filenames = files[args[:10]]
    
    print("Copie des annonces")
    nameDir = "../predicted"
    if not os.path.exists(nameDir):
        os.makedirs(nameDir)
    k=0   
    for fname in filenames:
        print(fname)
        shutil.copy(fname,nameDir+"/%d"%(k))
        k+=1
    print("Annonces selectionnées disponibles dans le dossier %s"%(nameDir))
    
if __name__ == "__main__":
   main(sys.argv[1])
   #main()