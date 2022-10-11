import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/allData.csv")
df.drop(columns=df.columns[0], axis=1, inplace=True)
df.drop(columns=df.columns[-1], axis=1, inplace=True)


