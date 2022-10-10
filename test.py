import pandas as pd
import numpy as np

df = pd.read_csv('./test_set.csv')
priceList = pd.read_csv('./sample_submission.csv')['price'].tolist()
# print(priceList)
df.reset_index


for index, row in df.iterrows():
    df.loc[index, 'price'] = priceList[df.loc[index, 'hotel_id']]
# print(df["price"].tolist)



df2=df.groupby(['city'])['price'].agg("mean")
print(df2)