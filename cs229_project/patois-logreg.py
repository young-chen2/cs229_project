import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Load the training dataset
train_df = pd.read_csv('datasets/patoisnli/jampatoisnli-train.csv')

# Load the test dataset
test_df = pd.read_csv('datasets/patoisnli/jampatoisnli-test.csv')

# Encode labels
label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# Text Data Preprocessing
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(train_df['premise'] + ' ' + train_df['hypothesis'])
X_train, X_dev, y_train, y_dev = train_test_split(X, train_df['label'], test_size=0.2, random_state=42)

# Lists to store accuracy for plotting
train_accuracy_list = []
dev_accuracy_list = []

# Experiment with different regularization strengths
regularization_strengths = [0.0001, 0.001, 0.01, 0.1, 1, 10]
test_accuracy_list = []

for strength in regularization_strengths:
    # Build the Logistic Regression Model
    model = LogisticRegression(C=strength, max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the dev data
    y_dev_pred = model.predict(X_dev)
    
    # Calculate Dev Accuracy
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    dev_accuracy_list.append(dev_accuracy)
    
    # Transform the test data using the same TF-IDF vectorizer
    X_test_tfidf = tfidf_vectorizer.transform(test_df['premise'] + ' ' + test_df['hypothesis'])
    
    # Predict on the test data
    y_test_pred = model.predict(X_test_tfidf)
    
    # Calculate Test Accuracy
    test_accuracy = accuracy_score(test_df['label'], y_test_pred)
    test_accuracy_list.append(test_accuracy)

    # Calculate Train Accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy_list.append(train_accuracy)

# Plot Train and Dev Accuracy
plt.plot(regularization_strengths, train_accuracy_list, label='Train Accuracy')
plt.plot(regularization_strengths, dev_accuracy_list, label='Dev Accuracy')
plt.xscale('log')
plt.xlabel('Regularization Strength')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Dev Accuracy vs Regularization Strength')
plt.savefig('train_dev_accuracy_plot_logreg.png')  # Save the plot as a .png file
plt.show()

# Plot Test Accuracy for different regularization strengths
plt.plot(regularization_strengths, test_accuracy_list, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Regularization Strength')
plt.savefig('test_accuracy_plot_logreg.png')  # Save the plot as a .png file
plt.show()

# Print Classification Report for the best model (highest dev accuracy)
best_strength = regularization_strengths[np.argmax(dev_accuracy_list)]
best_model = LogisticRegression(C=best_strength, max_iter=1000)
best_model.fit(X_train, y_train)
y_test_best_pred = best_model.predict(X_test_tfidf)
classification_rep_best = classification_report(test_df['label'], y_test_best_pred, target_names=label_mapping.keys())
print(f'Best Model (C={best_strength}) Classification Report:\n{classification_rep_best}')
