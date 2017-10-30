 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:16:43 2017

@author: akli
"""

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
import os.path
import codecs
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def getFile(path):
    with codecs.open(path) as f:
        return f.read()    

path = u"../Corpus"

languages = ['french', 'english', 'german', 'spanish']
stop_words = []
for l in languages:
    for w in stopwords.words(l):
        stop_words.append(w.encode("utf-8"))
        
alldocs = []
lab = 0
labels = []
for d in os.listdir(path):
    print(d)
    lab+=1
    for f in os.listdir(path+"/"+d):
        alldocs.append(getFile(path+"/"+d+"/"+f)) 
        labels.append(lab)
        
cv = getFile("../cv2.txt")
labels = np.asarray(labels)
#Transformation des documents en vecteurs sparses
vectorizer = TfidfVectorizer(max_df=1.0, min_df=2.*1./len(alldocs), stop_words=stop_words)
X = vectorizer.fit_transform(alldocs)
#Réduction de dimension
svd = TruncatedSVD(100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
#Transformation en vecteurs numeriques
data = lsa.fit_transform(X)
#Clustering sur les vecteurs numeriques et prediction d'un cluster pour chaque
#document de la base d'apprentissage
eps = np.linspace(.01,.1,10)
repartitions = []
purete = []
for e in eps : 
    db = DBSCAN(eps=e,metric='cosine',algorithm="brute")
    pred = db.fit_predict(data)
    p = []
    for cluster in np.unique(db.labels_):
        categoriesCluster = labels[np.where(pred == cluster)]
        compte = np.bincount(categoriesCluster)
        p.append(np.max(compte)*1.0/np.sum(compte))
    purete.append(p)
    print("eps = %f ; %d clusters"%(e,len(np.unique(db.labels_))))
    repartitions.append(np.bincount(pred+1))

"""
#Recuperation du cluster du CV
cvBow = vectorizer.transform([cv])
dataCV = lsa.transform(cvBow)
cluster = dbscan.labels_[np.argsort(cosine_similarity(data,dataCV))]
#Recuperation des indices des annonces correspondant au cluster
clusterData = np.where(prediction == cluster)[0]
closestAnnonces = np.argsort(cosine_similarity(data,dataCV).reshape((-1)))[len(data)-10:]
nameDir = u"predictedLSA"

if not os.path.exists(nameDir):
    os.makedirs(nameDir)
    
for findex in closestAnnonces:
    f = codecs.open(nameDir+"/%d.txt"%(findex),'w',"utf-8")
    #print alldocs[findex].encode("utf-8")
    f.write(alldocs[findex])
    f.close()
"""