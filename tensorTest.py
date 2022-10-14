import matplotlib.pyplot as plt
from matplotlib.streamplot import Grid
import numpy as np
import pandas as pd
from sklearn import metrics
import features
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor

'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''



# On récupère le dataFrame
df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))

# On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
df = df.drop(columns=["avatar_id"])

# On rajoute les attributs propre aux hôtels
df = features.prepareDataframe(df)

# on récupère la colonne cible, le prix, et on la supprime
y = df["price"]
df.drop(["price"], axis=1, inplace=True)


# On encode les données non numériques avec OneHotEncoder
columns_transfo = make_column_transformer(
    (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
    remainder='passthrough')
transformed = columns_transfo.fit_transform(df).toarray()
df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())


# On crée le jeu de tests et d'entraînement
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# On normalise les données
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

def create_model(layers, activation):
    model = keras.Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train_transformed.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    return model
 
model = KerasRegressor(build_fn=create_model, verbose=0)
print(model)

layers = [[20], [40, 20], [45, 30 ,15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


grid_result = grid.fit(X_train_transformed, y_train)

print(grid_result.best_score_, grid_result.best_params_)

print("\nTestAccuracy\n")

currentScore = mean_squared_error(y_test, grid.predict(X_test_transformed))
print("MSE score pour  --->  ", currentScore)