# P6: MULTIPLE REGRESSION MODEL 
# Apply multiple regressions, if data have a continuous Independent variable. Apply on above dataset. 

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('prac6/admission.csv')

# Name the dataset
df.name = "Admissions_Dataset"

# Define independent variables (X) and dependent variable (y)
X = df[['gre', 'gpa', 'rank']]  # Independent variables
y = df['admit']  # Dependent variable

# Adding constant term for intercept
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())
