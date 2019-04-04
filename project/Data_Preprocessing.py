#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import numpy as np
import ast
import json
import os
import sys
import re
from textblob import TextBlob
import collections
import pandas as pd
import requests
from urllib.parse import urlparse
from time import sleep


# In[49]:


def featuresExtraction(features_file):
    #convertedArray = json.load(open(features_file, 'r'))
    tweets = []
    for line in open(features_file, 'r'):
        tweets.append(json.loads(line))


    features = [['Text', 'Retweet count', 'Favorite count', 'Coordination']]
    for i, j in enumerate(tweets):
        feature = []
#         if 'retweeted_status'in j and 'extended_tweet' in j['retweeted_status']:
#             features.append(j['retweeted_status']['extended_tweet']['full_text'])
#         elif 'retweeted_status' not in j and 'extended_tweet' in j:
#             features.append(j['extended_tweet']['full_text'])
#         else:
#             features.append(j['text'])
        feature.append(j['text'].encode("utf-8"))
        if (j['retweet_count'] > 0):
            print ("Has retweets")
        feature.append(j['retweet_count'])
        feature.append(j['favorite_count'])
        feature.append(j['coordinates'])
        features.append(feature)
    
    return np.array(features)


# In[50]:


tweets = featuresExtraction('dataMining_190322.json')


# In[45]:


for i in range (25):
    print('Text: ', tweets[i][0])
    print('Retweet count: ', tweets[i][1])
    print('Favorite count: ', tweets[i][2])
    print('Coordinates: ', tweets[i][3])
    print()


# In[46]:


import csv
with open('test.csv', 'w', newline='') as out_f: # Python 3
    w = csv.writer(out_f, delimiter=',')        # override for tab delimiter
    w.writerows(tweets)                           # writerows (plural) doesn't need for loop


# In[5]:


import pandas as pd 
df = pd.read_csv("test.csv") 
df.head()


# In[13]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 
from wordcloud import WordCloud


# In[22]:


def partition(X, t):
    X_test = X[:int(t*len(X))]
    #y_test = y[:int(t*len(y))]
    X_train = X[int(t*len(X)):]
    #y_train = y[int(t*len(y)):]
    
    return X_train, X_test

X = df['Text']
X_other = X

# X = (X - X.mean())/X.std()

X_train, X_test = partition(X, 0.2)


# In[23]:


X_train = np.array(df["Text"])


# In[24]:


X_train[0]


# In[25]:


lemmatizer = WordNetLemmatizer()

X_train_lemmatized = []

for document in X_train:
    word_list = nltk.word_tokenize(document)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    X_train_lemmatized.append(lemmatized_output)


# In[26]:


X_train_lemmatized[0]


# In[27]:



from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


# In[28]:


#count_vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
#count_vect = CountVectorizer(lowercase=True, stop_words='english')
#count_vect = CountVectorizer(lowercase=True, stop_words='english', binary=True)
count_vect = CountVectorizer()


X_train_counts = count_vect.fit_transform(X_train_lemmatized)
print(X_train_counts.shape)

print("Type of the occurance count matrix (should be sparse): ")
print(type(X_train_counts))

X_test_counts = count_vect.transform(X_test)
print(X_test_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)
print(X_test_tfidf.shape)


# In[ ]:




