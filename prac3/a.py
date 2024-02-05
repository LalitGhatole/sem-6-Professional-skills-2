import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Iris dataset from CSV file
iris_file_path = 'prac3/iris.csv'  # Update with the correct file path
df_iris = pd.read_csv(iris_file_path)

# Set the style of seaborn for better aesthetics
sns.set(style="whitegrid")

# Box plot for each numeric column with respect to 'variety'
plt.figure(figsize=(14, 8))
for i, col in enumerate(['sepal.length', 'sepal.width']):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(x='variety', y=col, data=df_iris)
    plt.title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()

# Scatter plot matrix
plt.figure(figsize=(12, 8))
sns.pairplot(df_iris, hue="variety", markers=["o", "s", "D"])
plt.suptitle('Scatter Plot Matrix - Iris Dataset', y=1.02)
plt.show()
