import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import features
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def regression():

    # On récupère le dataFrame
    df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))
    # df = df.drop_duplicates(subset=['city', 'language', "hotel_id", "date"], keep='last')
    df = features.prepareDataframe(df)
    y = df["price"]
    df.drop(["price"], axis=1, inplace=True)
    # print(df.shape)

    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())
    df.to_csv('testaftertransform.csv')

    # On crée le jeu de tests et d'entraînement
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # for i in np.linspace(16, 1024, 5, dtype=int):
    #     for j in np.linspace(16, 1024, 5, dtype=int):
    reg = MLPRegressor(hidden_layer_sizes=(2,2,2,2), activation="relu",
                                random_state=1, max_iter=3000).fit(X_train, y_train)
    currentScore = mean_squared_error(y_test, reg.predict(X_test))
    print("| score = ", currentScore)



if __name__=="__main__":
    regression()