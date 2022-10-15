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
tabX= []
tabY = []
for avatID in avID:
    tabX.append(features.getAvatarName(avatID))
    tabY.append(features.getMinDayOfAvatar(features.getAvatarName(avatID)))

# Horizontal Bar Plot
fig = px.bar(x=tabX, y=tabY)
st.plotly_chart(fig)
fig.show()