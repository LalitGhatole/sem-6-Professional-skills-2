# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv('https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/admission.csv')

# Name your dataset
df.name = "Admissions_Dataset"

# Checking the first few rows of the dataset
print(df.head())

# Separating features and target variable
X = df[['gre', 'gpa', 'rank']]
y = df['admit']

# Adding constant term for logistic regression
X = sm.add_constant(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting logistic regression model
log_reg = sm.Logit(y_train, X_train)
result = log_reg.fit()

# Checking model summary
print(result.summary())

# Predicting on the test set
y_pred = result.predict(X_test)

# Converting probabilities to binary predictions
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy of the logistic regression model:", accuracy)