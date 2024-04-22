import pandas as pd

data = pd.read_csv('archive/data.csv')
data.fillna("", inplace=True)
data.drop(columns=[data.columns[0], 'Cleaned_Ingredients'], inplace=True)
data['Rating'] = 50