import tensorflow as tf

from tensorflow import keras
from keras import layers
import pandas as pd

print(tf.__version__)


def etcestparti():

    df = pd.read_csv('ceciestuntest.csv')
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

