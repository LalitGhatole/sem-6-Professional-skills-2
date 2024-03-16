# P7: CLASSIFICATION MODEL 
# a. Install relevant package for classification. 
# b. Choose classifier for classification problem. 
# c. Evaluate the performance of classifier. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# b. Choose classifier for classification problem
classifier = LogisticRegression()

df = pd.read_csv('prac7/iris.csv')

# Name the dataset
df.name = "Classification_Dataset"

# Define features (X) and target variable (y)
X = df.drop('variety', axis=1)  # Features
y = df['variety']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(solver='saga')
# Fit the classifier on the training data
classifier.fit(X_train, y_train)

# Predict on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
