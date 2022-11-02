import matplotlib.pyplot as plt
import pandas as pd
import features
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error

'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''


# On récupère le dataFrame
df = features.prepareDataframe(pd.read_csv("./data/allData.csv"))
df.drop(["Unnamed: 0"], axis=1, inplace=True)

print(df.dtypes)

# On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
df = df.drop(columns=["avatar_id"])

print("test2")

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

print("test")

# On normalise les données
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

def build_and_compile_model():
    print("ca pete la")
    model = keras.Sequential([
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model()
dnn_model.summary()


history = dnn_model.fit(
    X_train_transformed,
    y_train,
    verbose=0, epochs=100)

plot_loss(history)