import pandas as pd

# Read a CSV file from the web
url_csv = 'https://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a851711e20c5eb28f9d54742d1fe2dc/datasets.csv'
df_csv_web = pd.read_csv(url_csv)

# Read a TXT file from the web
url_txt = 'https://raw.githubusercontent.com/selva86/datasets/master/sample.txt'
df_txt_web = pd.read_csv(url_txt, delimiter='\t')  # Assuming tab-separated data

# Read a CSV file from disk
file_csv_disk = 'prac2/iris.csv'
df_csv_disk = pd.read_csv(file_csv_disk)

# Read a TXT file from disk
file_txt_disk = 'prac2/sample.txt'
df_txt_disk = pd.read_csv(file_txt_disk, delimiter='\t')  # Assuming tab-separated data

# Concatenate the dataframes
combined_df = pd.concat([df_csv_web, df_txt_web, df_csv_disk, df_txt_disk], ignore_index=True)

# Display the shape (number of rows and columns) of the combined dataframe
print("Shape of the combined dataframe:", combined_df.shape)

# Write the combined dataframe to a file on disk
output_file = 'prac2/combined_data.csv'
combined_df.to_csv(output_file, index=False)

print("Combined data has been written to:", output_file)
