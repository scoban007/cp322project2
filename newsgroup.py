from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# fetch dataset 
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
remove = ('headers', 'footers', 'quotes')   #FYI - Project says to remove these so the numbers are different from tutorial

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)

# Tutorial: loading dataset 
# print(twenty_train.target_names)
# print(len(twenty_train.data))
# print(len(twenty_train.filenames))
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(twenty_train.target_names[twenty_train.target[0]])
# twenty_train.target[:10]
# for t in twenty_train.target[:10]:
#     print(twenty_train.target_names[t])


#Extracting Features from text files - Bag of Words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)

#print(count_vect.vocabulary_.get(u'algorithm'))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)