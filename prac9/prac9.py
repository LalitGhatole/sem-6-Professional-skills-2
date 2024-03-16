# P9: Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and incorrect predictions.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('prac9/iris.csv')

# Define features (X) and target variable (y)
X = df.drop('variety', axis=1)  # Features
y = df['variety']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier on the training data
knn.fit(X_train, y_train)

# Predict on the testing data
y_pred = knn.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print correct and incorrect predictions
correct_predictions = X_test[y_test == y_pred]
incorrect_predictions = X_test[y_test != y_pred]

print("\nCorrect Predictions:")
print(correct_predictions)
print("\nIncorrect Predictions:")
print(incorrect_predictions)
