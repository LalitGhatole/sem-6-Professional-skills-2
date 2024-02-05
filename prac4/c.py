import pandas as pd
from scipy.stats import f_oneway

# Read the Iris dataset from CSV file
iris_file_path = 'prac4/iris.csv'
df_iris = pd.read_csv(iris_file_path)

# Perform ANOVA for each numeric column with respect to the 'variety' column
anova_results = {}
numeric_columns = df_iris.select_dtypes(include=['float64']).columns

for col in numeric_columns:
    groups = [df_iris[col][df_iris['variety'] == variety] for variety in df_iris['variety'].unique()]
    anova_result = f_oneway(*groups)
    anova_results[col] = anova_result

# Display the ANOVA results
for col, result in anova_results.items():
    print(f"ANOVA for {col}: F-statistic={result.statistic}, p-value={result.pvalue}")
