import pandas as pd

# Read the Iris dataset from CSV file
iris_file_path = 'prac4/iris.csv'
df_iris = pd.read_csv(iris_file_path)

# Exclude non-numeric columns (e.g., 'variety') before calculating correlation
numeric_columns = df_iris.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df_iris[numeric_columns]

# Display the correlation matrix
correlation_matrix = df_numeric.corr()
print("Correlation Matrix:")
print(correlation_matrix)
