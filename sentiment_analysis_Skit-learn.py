# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:29:31 2020

@author: samsung
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


import pickle
from sklearn.linear_model import LogisticRegressionCV

def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|p)',text)
    text=re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

#Tokenization of documents
porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
stop=stopwords.words('english')

#input data
df=pd.read_csv('movie_data.csv')

#preprocessing the review 
df['review']=df['review'].apply(preprocessor)

#Transforming text data into tf-idf vector
tfidf= TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=None,
                       tokenizer=tokenizer_porter,
                       use_idf=True,
                       norm='l2',
                       smooth_idf=True)
y=df.sentiment.values
x=tfidf.fit_transform(df.review)

#Document Classification Using Logestic Regression
#spliting the training and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,
                                               test_size=0.5,shuffle=False)

#Logistic Regression Model
clf=LogisticRegressionCV(cv=5,
                       scoring='accuracy',
                       random_state=0,
                       n_jobs=-1,
                       verbose=3,
                       max_iter=300).fit(x_train,y_train)

#saving the model
save_model=open('saved_model.sav','wb')
pickle.dump(clf,save_model)
save_model.close()

#testing the accuracy
filename= 'saved_model.sav'

n=clf.score(x_test,y_test)

print('The Accuracy of the model:')
print(n)

