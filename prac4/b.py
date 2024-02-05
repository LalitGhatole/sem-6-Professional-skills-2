import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Iris dataset from CSV file
iris_file_path = 'prac4/iris.csv'
df_iris = pd.read_csv(iris_file_path)

# Exclude non-numeric columns (e.g., 'variety') before calculating correlation
numeric_columns = df_iris.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df_iris[numeric_columns]

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heatmap - Iris Dataset')
plt.show()
