import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/sachu/Desktop/DB/database.csv')
print(read)
num_splits = 10
dfs = np.array_split(df, num_splits)

# Save each split DataFrame to a separate CSV file
for i, df_split in enumerate(dfs):
    file_name = f'db{i+1}.csv'  # Naming files as db1.csv, db2.csv, ..., db10.csv
    df_split.to_csv(file_name, index=False)
    print(f'Saved {file_name} with {len(df_split)} rows.')


print("done")