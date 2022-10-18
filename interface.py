import streamlit as st
import features
import numpy as np
import random


st.set_page_config(layout="wide")

def featureSelection():
    featuresImplemented = ["Dis moi bg", "Request generation", "Plotting", "Create avatar"]
    choice = st.sidebar.selectbox("Tu veux faire quoi maggle ?", featuresImplemented)

    if choice == "Request generation":
        features.stGenRequest()
    if choice == "Plotting":
        features.stPlotting()
    if choice == "Create avatar":
        features.stCreateAvatar()


if __name__=="__main__":
    featureSelection()