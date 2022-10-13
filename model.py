import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import features
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from numba import jit, cuda
from sklearn.ensemble import RandomForestRegressor



def regression():

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
    
    print(" --- Liste des colonnes, dataset train  ---")
    print(df.columns)

    # On crée le jeu de tests et d'entraînement
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


    # On normalise les données
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
 
        
    ## Création du modèle !

    # Meilleur Score :  i =  128 j =  32

    minScore = 4000
    max_depth = 0

    for i in range(29, 41):
        clf = RandomForestRegressor(max_depth=i, min_samples_leaf=1, random_state=0).fit(X_train_transformed, y_train)
        currentScore = mean_squared_error(y_test, clf.predict(X_test_transformed))
        ##print("MSE score pour i = ", i, "  --->  ", currentScore)
        if currentScore< minScore:
            minScore = currentScore
            max_depth = i           

    print("\nRésultat trouvé : max depth = ", max_depth)
    print("\nAvec un score MSE = ", minScore)

    
    return clf


if __name__=="__main__":
    regression()