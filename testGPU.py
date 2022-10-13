import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import features
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# On récupère le dataFrame
df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))

# On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
df = df.drop(columns=["avatar_id"])

# On rajoute les attributs propre aux hôtels
df = features.prepareDataframe(df)

# on récupère la colonne cible, le prix, et on la supprime
y = df["price"]
df.drop(["price"], axis=1, inplace=True)

print(df)

# On encode les données non numériques avec OneHotEncoder
columns_transfo = make_column_transformer(
    (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
    remainder='passthrough')
transformed = columns_transfo.fit_transform(df).toarray()
df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())

print(df)

# On crée le jeu de tests et d'entraînement
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# On normalise les données
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

print(len(X_train_transformed[0]))

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train_transformed[0])]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mae', 'mse'])

