#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:48:12 2021

@author: asaap
"""

import gradio as gr
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer


model = pickle.load(open('spam_classifier.pkl', 'rb'))
cv = pickle.load(open('count_vect', 'rb'))


def spam_classifier(message):
    ps = PorterStemmer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    X = cv.transform(corpus).toarray()
    
    prediction = model.predict(X)
    if prediction==1:
        return 'SPAM!'
    else:
        return 'HAM.'

iface = gr.Interface(
  fn=spam_classifier, 
  inputs=gr.inputs.Textbox(lines=2, placeholder="Paste Your Message Here..."), 
  outputs="text")
iface.launch()
