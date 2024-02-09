import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load the Iris dataset
iris_df = pd.read_csv("prac4/iris.csv")

# Display the first few rows of the dataset
print(iris_df.head())

# a. Correlation matrix (excluding the categorical column 'variety')
correlation_matrix = iris_df.drop('variety', axis=1).corr()
print("Correlation Matrix:")
print(correlation_matrix)

# b. Correlation plot
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Plot of Iris Dataset")
plt.show()

# c. Analysis of covariance (ANOVA)
# Perform ANOVA on petal lengths for each variety
setosa_petal_length = iris_df[iris_df['variety'] == 'Setosa']['petal.length']
versicolor_petal_length = iris_df[iris_df['variety'] == 'Versicolor']['petal.length']
virginica_petal_length = iris_df[iris_df['variety'] == 'Virginica']['petal.length']

# Perform ANOVA
anova_result = f_oneway(setosa_petal_length, versicolor_petal_length, virginica_petal_length)
print("ANOVA Result:")
print(anova_result)
