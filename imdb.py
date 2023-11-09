import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import time
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def data_fetch(path, label_map=None):
    texts = []
    labels = []
    
    if label_map is None:
        label_map = {'pos': 1, 'neg': 0}

    for label in label_map:
        dir_path = os.path.join(path, label)

        for file in os.listdir(dir_path):
            with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label_map[label])
    
    return texts, labels

def preprocessing(text):
    #Pre-processing dataset: use only lowercase and remove trailing white spaces, 
    # removing common words like "the" "is" etc, 
    # removing puncuations, lemmatization - using root words? 
    
    #Initialize the NLTK stopwords and WordNet lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    #Convert text to lowercase and remove traillng spaces
    text = text.lower().strip()

    #Remove punctuation using regular expressions
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)

    #Tokenize the text
    tokens = text.split()

    # Remove common words (stop words)
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization (using WordNetLemmatizer)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def main():
        
    print("Loading training data...")
    train_text, train_label = data_fetch('imdb/aclImdb/train')
    print("Loading test data...")
    test_text, test_label = data_fetch('imdb/aclImdb/test')

    # print("Number of training samples:", len(train_text))
    # print("Number of test samples:", len(test_text))

    #Vectorizing and using preprocessing method 
    print("Vectorizing & preprocessing data...")
    vectorizer = CountVectorizer(preprocessor=preprocessing)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    #Models with specific parameters 
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10),
        "SVM": LinearSVC(dual=False, max_iter=5000),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10)
    }

    #Train, predict, and get score for each model
    for name, model in models.items():

        start_time = time.time()

        print(f"Training {name}...")
        model.fit(X_train, train_label)

        print(f"Predicting with {name}...")
        y_pred = model.predict(X_test)

        end_time = time.time()
        elapsed_time = end_time - start_time

        acc = accuracy_score(test_label, y_pred)
        
        print(f"{name} Accuracy: {acc}")
        print(f"Elapsed Time: {elapsed_time: .2f} seconds.")


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform k-fold cross-validation for each model
    for name, model in models.items():
        
        print(f"Validating {name} using k-fold cross-validation...")

        start_time = time.time()

        # Perform cross-validation and get accuracy scores
        scores = cross_val_score(model, X_train, train_label, cv=kfold, scoring='accuracy')

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print average accuracy and standard deviation
        print(f"{name} Average Accuracy: {np.mean(scores):.4f}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds.")

    #Results for reference:
    # Loading training data...
    # Loading test data...
    # Vectorizing & preprocessing data...
    # Training Logistic Regression...
    # Predicting with Logistic Regression...
    # Logistic Regression Accuracy: 0.86448
    # Elapsed Time:  7.09 seconds.
    # Training Decision Tree...
    # Predicting with Decision Tree...
    # Decision Tree Accuracy: 0.7292
    # Elapsed Time:  7.94 seconds.
    # Training SVM...
    # Predicting with SVM...
    # SVM Accuracy: 0.84624
    # Elapsed Time:  36.50 seconds.
    # Training AdaBoost...
    # Predicting with AdaBoost...
    # AdaBoost Accuracy: 0.80264
    # Elapsed Time:  21.99 seconds.
    # Training Random Forest...
    # Predicting with Random Forest...
    # Random Forest Accuracy: 0.8082
    # Elapsed Time:  2.74 seconds.
    # Validating Logistic Regression using k-fold cross-validation...
    # Logistic Regression Average Accuracy: 0.8795
    # Elapsed Time: 26.48 seconds.
    # Validating Decision Tree using k-fold cross-validation...
    # Decision Tree Average Accuracy: 0.7213
    # Elapsed Time: 46.20 seconds.
    # Validating SVM using k-fold cross-validation...
    # SVM Average Accuracy: 0.8638
    # Elapsed Time: 254.18 seconds.
    # Validating AdaBoost using k-fold cross-validation...
    # AdaBoost Average Accuracy: 0.7982
    # Elapsed Time: 725.99 seconds.
    # Validating Random Forest using k-fold cross-validation...
    # Random Forest Average Accuracy: 0.8063
    # Elapsed Time: 25.90 seconds.

if __name__ == "__main__":
    main()