from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#Fetching dataset 
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
remove = ('headers', 'footers', 'quotes')   #FYI - Project says to remove these so the numbers are different from tutorial

print("Fetching data...")
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)

#Tutorial: loading dataset 
# print(twenty_train.target_names)
# print(len(twenty_train.data))
# print(len(twenty_train.filenames))
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(twenty_train.target_names[twenty_train.target[0]])
# twenty_train.target[:10]
# for t in twenty_train.target[:10]:
#     print(twenty_train.target_names[t])


#Extracting Features from text files - Bag of Words
#Vectorizing
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)

#print(count_vect.vocabulary_.get(u'algorithm'))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)

#Select samplesize 
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Models with specific parameters 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "SVM": LinearSVC(dual=False),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}


for name, model in models.items():
    print(f"Training and predicting {name}...")
    model.fit(X_train, y_train)
    #print(f"Predicting with {name}...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}\n")


#Results for reference:
# Training and predicting Logistic Regression...
# Logistic Regression Accuracy: 0.895

# Training and predicting Decision Tree...
# Decision Tree Accuracy: 0.9

# Training and predicting SVM...
# SVM Accuracy: 0.895

# Training and predicting AdaBoost...
# AdaBoost Accuracy: 0.92

# Training and predicting Random Forest...
# Random Forest Accuracy: 0.925