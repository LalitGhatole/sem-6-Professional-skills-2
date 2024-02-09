import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris_df = pd.read_csv("prac3/iris.csv")

# Display the first few rows of the dataset
print(iris_df.head())

# a. Box and scatter plots for data distributions
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df)
plt.title("Box Plot of Iris Dataset")
plt.xlabel("Features")
plt.ylabel("Centimeters")
plt.xticks(rotation=45)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal.length', y='sepal.width', hue='variety')
plt.title("Scatter Plot of Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# b. Finding outliers using box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df)
plt.title("Box Plot to Identify Outliers")
plt.xlabel("Features")
plt.ylabel("Centimeters")
plt.xticks(rotation=45)
plt.show()

# c. Histogram and pie chart
# Histogram
plt.figure(figsize=(10, 6))
plt.hist(iris_df['petal.length'], bins=20, edgecolor='black')
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Pie chart
variety_counts = iris_df['variety'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(variety_counts, labels=variety_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Iris variety")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
