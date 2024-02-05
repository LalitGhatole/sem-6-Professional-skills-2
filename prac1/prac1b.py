import pandas as pd

# Load the Iris dataset from CSV file
iris_df = pd.read_csv('prac1/iris.csv')

# Display a subset of the dataset using subset() function
subset_condition = (iris_df['sepal.length'] > 5) & (iris_df['sepal.width'] > 3)
subset_result = iris_df[subset_condition]

print("Subset of the Iris dataset using subset() function:")
print(subset_result)
print("\n")

# Aggregate the dataset using aggregate() function
aggregation_result = iris_df.groupby('variety').aggregate({
    'sepal.length': 'mean',
    'sepal.width': 'std',
    'petal.length': 'mean',
    'petal.width': 'std'
})
aggregation_result.rename(columns={
    'sepal.length': 'mean_sepal_length',
    'sepal.width': 'std_sepal_width',
    'petal.length': 'mean_petal_length',
    'petal.width': 'std_petal_width'
}, inplace=True)

print("Aggregation of the Iris dataset using aggregate() function:")
print(aggregation_result)
