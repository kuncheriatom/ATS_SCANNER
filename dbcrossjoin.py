import pandas as pd
import numpy as np


db =  pd.read_csv('C:/Users/sachu/Desktop/DB/database.csv')

db.head()

db.info()


db = db.drop(columns=['Resume_html'])


# # Load the CSV files
resume_df = pd.read_excel("C:/Users/sachu/Desktop/DB/resume.xlsx")
posting_df = pd.read_csv('C:/Users/sachu/Desktop/DB/postings.csv')

#  Randomly select 1000 rows from each table
resume_sample = resume_df.sample(n=1000, random_state=1)
posting_sample = posting_df.sample(n=100, random_state=1)

#  Add a key column to each dataframe to facilitate the cross join
resume_sample['key'] = 1
posting_sample['key'] = 1

# # Perform the cross join
cross_join_df = pd.merge(resume_sample, posting_sample, on='key').drop('key', axis=1)

# # Save the result to a new CSV file
cross_join_df.to_csv('database.csv', index=False)

print("Cross join completed and saved to 'database.csv'")

