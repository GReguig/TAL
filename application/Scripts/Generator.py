#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:43:20 2017

@author: akli
"""

import codecs
import os.path
import re

class DataGen(object):
    
    def __init__(self,pathDir,stopWords = False):
        self.pathDir = pathDir
        self.punctuation = '.!\"#?$%&(")*,-:;<=>@[]^_`{|}~*'
        self.maketrans = str.maketrans('', '', self.punctuation)    
        self.stopWords = stopWords
        if(stopWords):
            from nltk.corpus import stopwords
            languages = ['french', 'english', 'german', 'spanish']
            self.stop_words = []
            for l in languages:
                for w in stopwords.words(l):
                    self.stop_words.append(w)

        
    def __iter__(self):
        
        for d in os.listdir(self.pathDir):
            print(d)
            for f in os.listdir(self.pathDir+"/"+d):
                fp = self.pathDir+"/"+d+"/"+f
                with codecs.open(fp,encoding="utf-8") as content:
                    #annonce = re.split("\.w|\\n|!|\?",content.read().lower())
                    #sp = phrase.translate(self.maketrans).replace("\'"," ").split()
                    annonce = re.split("\.|\\n|\?|!",content.read().lower())
                    for phrase in annonce:
                        sp = phrase.translate(self.maketrans).replace("\'"," ").split()
                        if self.stopWords:
                            yield [i for i in sp if i not in self.stop_words]
                        else : 
                            yield sp
class DocGen(object):
    
    def __init__(self,pathDir):
        self.pathDir = pathDir
        self.punctuation = '.!\"#?$%&(")*,-:;<=>@[]^_`{|}~*'
        self.maketrans = str.maketrans('', '', self.punctuation)
        self.categories = []
        self.files = []
        
    def __iter__(self):
        del self.categories[:]
        del self.files[:]
        label = -1
        for d in os.listdir(self.pathDir):
            print(d)
            label+=1
            for f in os.listdir(self.pathDir+"/"+d):    
                fp = self.pathDir+"/"+d+"/"+f
                self.files.append(fp)
                with codecs.open(fp,encoding="utf-8") as content:
                    #annonce = re.split("\.w|\\n|!|\?",content.read().lower())
                    #annonce = re.sub(string.punctuation," ",content.read().lower()).replace("\'"," ")
                    annonce = content.read().lower().translate(self.maketrans).replace("\'"," ").split()
                    self.categories.append(label)
                    yield annonce
                        