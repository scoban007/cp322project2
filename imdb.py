import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

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

print("Loading training data...")
train_text, train_label = data_fetch('imdb/aclImdb/train')
print("Loading test data...")
test_text, test_label = data_fetch('imdb/aclImdb/test')

print("Number of training samples:", len(train_text))
print("Number of test samples:", len(test_text))

print("Vectorizing...")
vectorizer = CountVectorizer()
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
    print(f"Training {name}...")
    model.fit(X_train, train_label)

    print(f"Predicting with {name}...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(test_label, y_pred)
    print(f"{name} Accuracy: {acc}")

#Results for reference:
# Loading training data...
# Loading test data...
# Number of training samples: 25000
# Number of test samples: 25000
# Vectorizing...
# Training Logistic Regression...
# Predicting with Logistic Regression...
# Logistic Regression Accuracy: 0.86676
# Training Decision Tree...
# Predicting with Decision Tree...
# Decision Tree Accuracy: 0.72408
# Training SVM...
# Predicting with SVM...
# SVM Accuracy: 0.84612
# Training AdaBoost...
# Predicting with AdaBoost...
# AdaBoost Accuracy: 0.80516
# Training Random Forest...
# Predicting with Random Forest...
# Random Forest Accuracy: 0.8136