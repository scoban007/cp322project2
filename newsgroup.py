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
from sklearn.model_selection import train_test_split, StratifiedKFold
import time

def main():
        
    #Fetching dataset 
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    remove = ('headers', 'footers', 'quotes')   #FYI - Project says to remove these so the numbers are different from tutorial

    print("Fetching data...")
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)

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
    X, y = make_classification(n_samples=2000, n_features=20)
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

        start_time = time.time()

        print(f"Training and predicting {name}...")

        model.fit(X_train, y_train)
        #print(f"Predicting with {name}...")
        y_pred = model.predict(X_test)

        end_time = time.time()
        elapsed_time = end_time - start_time
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc}")
        print(f"Elapsed Time: {elapsed_time: .3f} seconds.\n")


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Perform k-fold cross-validation for each model
    for name, model in models.items():

        start_time = time.time()

        print(f"Validating {name} using k-fold cross-validation...")

        # Initialize accuracy scores list for each fold
        accuracy_scores = []

        for train_indices, test_indices in kfold.split(X_train_tfidf, twenty_train.target):
            X_train_fold, X_test_fold = X_train_tfidf[train_indices], X_train_tfidf[test_indices]
            y_train_fold, y_test_fold = twenty_train.target[train_indices], twenty_train.target[test_indices]

            # Train and predict for each fold
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)

            acc = accuracy_score(y_test_fold, y_pred)
            accuracy_scores.append(acc)

        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print average accuracy and standard deviation
        print(f"{name} Average Accuracy: {np.mean(accuracy_scores):.4f}")
        print(f"Elapsed Time: {elapsed_time:.3f} seconds.\n")

    # Fetching data...
    # Training and predicting Logistic Regression...
    # Logistic Regression Accuracy: 0.9175    
    # Elapsed Time:  0.007 seconds.

    # Training and predicting Decision Tree...
    # Decision Tree Accuracy: 0.9125
    # Elapsed Time:  0.083 seconds.      

    # Training and predicting SVM...     
    # SVM Accuracy: 0.92
    # Elapsed Time:  0.002 seconds.      

    # Training and predicting AdaBoost...
    # AdaBoost Accuracy: 0.9025
    # Elapsed Time:  0.682 seconds.

    # Training and predicting Random Forest...
    # Random Forest Accuracy: 0.935
    # Elapsed Time:  1.450 seconds.

    # Validating Logistic Regression using k-fold cross-validation...
    # Logistic Regression Average Accuracy: 0.841
    # Elapsed Time: 5.11 seconds.

    # Validating Decision Tree using k-fold cross-validation...
    # Decision Tree Average Accuracy: 0.534
    # Elapsed Time: 5.10 seconds.

    # Validating SVM using k-fold cross-validation...
    # SVM Average Accuracy: 0.876
    # Elapsed Time: 0.67 seconds.

    # Validating AdaBoost using k-fold cross-validation...
    # AdaBoost Average Accuracy: 0.649
    # Elapsed Time: 41.49 seconds.

    # Validating Random Forest using k-fold cross-validation...
    # Random Forest Average Accuracy: 0.747
    # Elapsed Time: 75.12 seconds.

if __name__ == "__main__":
    main()