#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:06:18 2019

@author: akm
"""

import sys
import nltk
import sklearn
import pandas
import numpy
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Code used for the Data Cleansing activity
def data_cleanse(main_text_messages):
    # Data Cleanse of the data 

    # Substitute email address with 'emailids'
    main_text_messages = text_messages.str.replace(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+','emailIDs')
    
    # Substitute of the URL or the Web Address
    main_text_messages = main_text_messages.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                      'URLs')
    
    #Substitute numbers with the "number" string
    main_text_messages = main_text_messages.str.replace(r'\d+(\.\d+)?', 'number')
    
    #Substitute phone number with "phone number"
    main_text_messages = main_text_messages.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')
    
    # Substitute money symbols with 'moneysymbol'
    main_text_messages = main_text_messages.str.replace(r'£|\$', 'moneysymbol')
    
    # Substitute punctuation
    main_text_messages = main_text_messages.str.replace(r'[^\w\d\s]', ' ')
    
    # Substitute whitespace between terms with a single space
    main_text_messages = main_text_messages.str.replace(r'\s+', ' ')
    
    # Substitute leading and trailing whitespace
    main_text_messages = main_text_messages.str.replace(r'^\s+|\s+?$', '')
    
    # convert all to one case (Higher/Lower) case
    main_text_messages = main_text_messages.str.lower()
    return main_text_messages

# Removal of Stopwords and Lemmatization
def stop_lemmatize(messages_to_be_cleansed):
    stop_words = set(stopwords.words('english'))
    messages_to_be_cleansed = messages_to_be_cleansed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    ps = nltk.PorterStemmer()

    messages_to_be_cleansed = messages_to_be_cleansed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))
    return messages_to_be_cleansed
    
# Tokenization
def tokenize(lemmatized_words):
    words = []
    
    for message in lemmatized_words:
        Twords = word_tokenize(message)
        for w in Twords:
            words.append(w)
            
    words = nltk.FreqDist(words)
    return words

# Finding the features
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in most_common:
        features[word] = (word in words)
    return features

# Classifier for the Naive Bayes 
def NBClassifier(training,testing):
    model = SklearnClassifier(MultinomialNB())
    # train the model on the training data
    model.train(training)
    # and test on the testing dataset!
    accuracy = nltk.classify.accuracy(model, testing)*100
    print("NB Accuracy: {}".format(accuracy))
    
# Classifier for DTree with Maximum entropy
def DTree(training,testing):
    model = SklearnClassifier(DecisionTreeClassifier())
    # train the model on the training data
    model.train(training)
    # and test on the testing dataset!
    accuracy = nltk.classify.accuracy(model, testing)*100
    print("DTree Accuracy with maximum entropy: {}".format(accuracy))
    
# Classifier for Logistic Regression a.k.a Maximum entropy
def LRClassification(training,testing):
    model = SklearnClassifier(LogisticRegression(solver='lbfgs'))
    # train the model on the training data
    model.train(training)
    # and test on the testing dataset!
    accuracy = nltk.classify.accuracy(model, testing)*100
    print("Logistic Regression AKA MaxEnt: {}".format(accuracy))
    
# load the dataset of SMS messages
# opening a file handler for reading a csv file contents into the memory for processing 
filehandler = pandas.read_csv('SMSSPamCollection',sep="\t", header=None, encoding='utf-8')
classes = filehandler[0]
XEncoder = LabelEncoder()
YEncoder = XEncoder.fit_transform(classes)
text_messages = filehandler[1]

# Calling a function for the Data Cleanse
cleansed_messages = data_cleanse(text_messages)
# Calling a function for the Stop-words and Lemmatization
lemmatized_messages = stop_lemmatize(cleansed_messages)
# Tokenize words
tokenized_words = tokenize(lemmatized_messages)
# Get the 2500 most common words
most_common = list(tokenized_words.keys())[:3000]
# call find features function
# features_list = find_feature_set(most_common,lemmatized_messages)
features_list = find_features(lemmatized_messages[0])
# print(features_list)

# Prepare the data for training and testing the classifier
processedMessages = zip(lemmatized_messages, YEncoder)
seed = 1
numpy.random.seed = seed
featuresets = [(find_features(text), label) for (text, label) in processedMessages]
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
print("Training Dataset: "+str(len(training)))
print("Testing Dataset:"+str(len(testing)))

# Call classifier for NB
NBClassifier(training , testing)
DTree(training , testing)
LRClassification(training, testing)