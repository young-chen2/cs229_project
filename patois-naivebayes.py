import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the training dataset
train_df = pd.read_csv('datasets/patoisnli/jampatoisnli-train.csv')

# Load the test dataset
test_df = pd.read_csv('datasets/patoisnli/jampatoisnli-test.csv')

# Encode labels
label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# Text Data Preprocessing - Bag of Words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(train_df['premise'] + ' ' + train_df['hypothesis'])
X_test_bow = vectorizer.transform(test_df['premise'] + ' ' + test_df['hypothesis'])

# Split the data into training and development sets
X_train, X_dev, y_train, y_dev = train_test_split(X_train_bow, train_df['label'], test_size=0.2, random_state=42)

# Lists to store accuracy for plotting
train_accuracy_list = []
dev_accuracy_list = []

# Experiment with different alpha values
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
test_accuracy_list = []

for alpha in alpha_values:
    # Build the Naive Bayes Classifier
    model = MultinomialNB(alpha=alpha)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Collect train accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    train_accuracy_list.append(train_accuracy)
    
    # Collect dev accuracy
    dev_accuracy = accuracy_score(y_dev, model.predict(X_dev))
    dev_accuracy_list.append(dev_accuracy)

    # Predict on the test data
    y_test_pred = model.predict(X_test_bow)
    
    # Calculate Test Accuracy
    test_accuracy = accuracy_score(test_df['label'], y_test_pred)
    test_accuracy_list.append(test_accuracy)

# Plot Train and Dev Accuracy
plt.plot(alpha_values, train_accuracy_list, label='Train Accuracy')
plt.plot(alpha_values, dev_accuracy_list, label='Dev Accuracy')
plt.xscale('log')
plt.xlabel('Alpha (Smoothing Parameter)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Dev Accuracy vs Alpha (Smoothing Parameter)')
plt.savefig('train_dev_accuracy_plot_nb.png')  # Save the plot as a .png file
plt.show()

# Plot Test Accuracy for different alpha values
plt.plot(alpha_values, test_accuracy_list, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (Smoothing Parameter)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Alpha (Smoothing Parameter)')
plt.savefig('test_accuracy_plot_nb.png')  # Save the plot as a .png file
plt.show()

# Print Classification Report for the best model (highest dev accuracy)
best_alpha = alpha_values[np.argmax(dev_accuracy_list)]
best_model = MultinomialNB(alpha=best_alpha)
best_model.fit(X_train_bow, train_df['label'])
y_test_best_pred = best_model.predict(X_test_bow)
classification_rep_best = classification_report(test_df['label'], y_test_best_pred, target_names=label_mapping.keys())
print(f'Best Model (Alpha={best_alpha}) Classification Report:\n{classification_rep_best}')
