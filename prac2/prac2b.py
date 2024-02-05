import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'prac2/airline.xlsx'

# Read the Excel file into a DataFrame
df_excel = pd.read_excel(excel_file_path)

# Display the DataFrame
print(df_excel)
