# -*- coding: utf-8 -*-
"""
"""

'''
import os
import codecs

from dataloadingLib import *
from unmaskingLib import *

from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

directories_compared = ['WithmanChunks','Questioned BDT 1857','allc1','allc2']
      

def graphComp():
 print("graphLib.graphComp")
 classifiers = [MultinomialNB(), LogisticRegression(),Perceptron(),svm.SVC(kernel='linear', C=1),KNeighborsClassifier(n_neighbors=3)]
 color_set = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
 
 
 for classifier in classifiers:
        print('---------------------')
        print(str(classifier)[:200])
        
        for directory_training in directories_compared:
            list_to_plot, i = [], 0
            f, ax = plt.subplots()
            
            for directory_target in directories_compared:
                
                x,y = [],[]
                if (directory_training is not directory_target):
                    data,target = getData(os.getcwd(),directory_training,directory_target)
                                       
                    for numfeatures in range(50,1001,200):
                            
                            #averaging_cvscores_number = 10                             
                            wordlist = getWords(data,numfeatures)                                  
                            scores= CVScores(data,target, wordlist, classifier, 10)                              
                            x.append(numfeatures), y.append(np.mean(scores))
                            
                    
                    ax.plot(x, y, color = color_set[i], label=directory_training+'/'+directory_target)
                    
                    plt.xlim(900, 0)
                    plt.ylim(0.5, 1.5)
                    #plt.ylim(0, 1.5)
                                     
                    i+=1
            # Now add the legend with some customizations.
            legend = ax.legend(loc='lower left', shadow=True)
            
            
            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            
            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width
                  
            plt.title("Unmasking Procedure")
            plt.xlabel("Number of features")
            plt.ylabel("Scores")
            plt.show(f)                          
        
            break
        
def graphCompWord():         
 classifier = MultinomialNB()
 color_set = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
  
 print('---------------------')
 print(str(classifier)[:20])
 
 for directory_training in directories_compared:
    list_to_plot, i = [], 0
    f, ax = plt.subplots()
    
    for directory_target in directories_compared:
        x,y = [],[]
        if (directory_training is not directory_target):
             data,target = getData(os.getcwd(),directory_training,directory_target)
             wordlist = getWords(data,500)
                
             for wordsRem in range(80):   
                    #averaging_cvscores_number = 10
                    scores = CVScores(data,target, wordlist, classifier, 10)
                    
                    x.append(len(wordlist)), y.append(np.mean(scores))
                                        
                    count_vect = CountVectorizer(vocabulary=wordlist)
                    X_train_counts = count_vect.fit_transform(data)
                    clf = classifier.fit(X_train_counts,target)    
                            
                    mostF = show_most_informative_features(count_vect,clf, 10)
                    wordlist.remove(mostF[0])
                    wordlist.remove(mostF[1])
                    
             ax.plot(x, y, color = color_set[i],label=directory_training+'/'+directory_target)
             plt.xlim(500,300)
             plt.ylim(0.7,1)
                       
             i+=1
             
    # Now add the legend with some customizations.
    legend = ax.legend(loc='lower left', shadow=True)
            
            
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')        
            
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
                  
    plt.title("Unmasking Procedure")
    plt.xlabel("Most informative features")
    plt.ylabel("Scores")
            
    plt.show(f)                          
        
    break               
                 
             
       
def graphCompWordTes():    
    classifiers = [MultinomialNB()]
    color_set = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
	
    for classifier in classifiers:
        
        print('---------------------')
        print(str(classifier)[:20])
        
                
        #x,y = [],[]
        for directory_training in directories_compared:
            list_to_plot, i = [], 0
            f, ax = plt.subplots()
            
            for directory_target in directories_compared:
                x,y = [],[]
                if (directory_training is not directory_target):
                    data,target = getData(os.getcwd(),directory_training,directory_target)
                    wordlist = getWords(data,500)
                    for wordsRem in range(80):  
                            #averaging_cvscores_number=10
                            scores= CVScores(data,target, wordlist, classifier, 10)
                            
                            #x.append(wordsRem), y.append(scores.mean())
                            x.append(len(wordlist)), y.append(np.mean(scores))
                            
                            count_vect = CountVectorizer(vocabulary=wordlist)
                            X_train_counts = count_vect.fit_transform(data)
                            clf = classifier.fit(X_train_counts,target)    
                            
                            mostF = show_most_informative_features(count_vect,clf, 1)
                            wordlist.remove(mostF[0])
                            wordlist.remove(mostF[1])
                    
                    ax.plot(x, y, color = color_set[i],label=directory_training+'/'+directory_target)
                    plt.xlim(500,300)
                    plt.ylim(0.7,1)
                    #ax.set_ylim(ymax=1)
                    #ax.set_ylim(ymax=0.6)
                    
                    i+=1
                    
                                            
            # Now add the legend with some customizations.
            legend = ax.legend(loc='lower left', shadow=True)
            
            
            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')        
                    
            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width
                          
            plt.title("Unmasking Procedure")
            plt.xlabel("Most informative features")
            plt.ylabel("Scores")
                    
            plt.show(f)                          
                
            break               
                 
    
def graphCompTes():
    classifiers = [MultinomialNB(),svm.SVC(kernel='linear', C=1),KNeighborsClassifier(n_neighbors=3)]
    color_set = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
    
    
    for classifier in classifiers:
        
        print('---------------------')
        print(str(classifier)[:20])
        
        for directory_training in directories_compared:
            list_to_plot, i = [], 0
            f, ax = plt.subplots()
            
            for directory_target in directories_compared:
                x,y = [],[]
                if (directory_training is not directory_target):
                    data,target = getData(os.getcwd(),directory_training,directory_target)
                    for numfeatures in range(50,1001,10):
                            #averaging_cvscores_number=10
                            
                            wordlist = getWords(data,numfeatures)   
                            scores= CVScores(data,target, wordlist, classifier, 10)                      
                            x.append(numfeatures), y.append((scores))
                                
                    
                    ax.plot(x, y, color = color_set[i], label=directory_training+'/'+directory_target)
                    plt.xlim(1000, 0)
                    plt.ylim(0.5,1.5)
                                                            
                    i+=1
                    
            # Now add the legend with some customizations.
            legend = ax.legend(loc='lower left', shadow=True)
            
            
            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            
            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width
                  
            plt.title("Unmasking Procedure")
            plt.xlabel("Number of features")
            plt.ylabel("Scores")
            plt.show(f)                          
        
            break

'''