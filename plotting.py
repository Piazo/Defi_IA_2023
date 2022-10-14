import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px
import features

# import plotly.express as px

df = pd.read_csv("./data/allData.csv")


# print(df)
avID=pd.unique(df["avatar_id"])
x= []
y = []
for avatID in avID:
    x.append(features.getAvatarName(avatID))
    y.append(features.getMinDayOfAvatar(features.getAvatarName(avatID)))
print(x, y)
# Horizontal Bar Plot
plt.bar(x, y)
 
# Show Plot
plt.show()