import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import features
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor

def regression():

    # On récupère le dataFrame
    df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))
    
    # On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
    df = df.drop(columns=["avatar_id", "request_order"])

    # On rajoute les attributs propre aux hôtels
    df = features.prepareDataframe(df)
    print(df)
    
    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"]
    df.drop(["price"], axis=1, inplace=True)

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
    
    print("X_train_transformed shape")

    ## Création du modèle !

    clf = RandomForestRegressor(max_depth=18, min_samples_leaf=1).fit(X_train_transformed, y_train)
    print("sans random : ", mean_squared_error(y_test, clf.predict(X_test_transformed)))
    
    clf = RandomForestRegressor(max_depth=18, min_samples_leaf=1, random_state=0).fit(X_train_transformed, y_train)
    print("avec random : ", mean_squared_error(y_test, clf.predict(X_test_transformed)))

    reg = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                                random_state=1, max_iter=2500).fit(X_train_transformed, y_train)
    currentScore = mean_squared_error(y_test, reg.predict(X_test_transformed))
    print("MSE score  = ", currentScore)

    
    minScore = 99999
    input1 = 0
    input2 = 0
    for i in [32, 64, 128, 256]:
        print(i)
        for j in [32, 64, 128, 256]:
            reg = MLPRegressor(hidden_layer_sizes=(i, j), activation="relu",
                                random_state=1, max_iter=2500).fit(X_train_transformed, y_train)
            currentScore = mean_squared_error(y_test, reg.predict(X_test_transformed))
            print("     ", j, " = ", currentScore)
            if(currentScore < minScore):
                minScore = currentScore
                input1 = i
                input2 = j
                  
    print(" i = ", input1)
    print(" j = ", input2)

    print("\nRésultat trouvé avec un score MSE = ", minScore)

    # Meilleur Score :  i =  128 j =  32


    # Use for the generation of the csv file
    
    '''
    header = ["index", "price"]
    data = []
    for i in range(len(test_data)):
        prediction = [i, int(clf.predict([test_data[i]]))]
        data.append(prediction)

    with open('testPredictions.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write data
        writer.writerows(data)
    '''













    '''
     # On veut minimiser MSE
    minScore = 4000
    max_depth = 0
    min_leaf = 0
    for i in range(1, 2):
        for j in range(1, 2):
            clf = RandomForestRegressor(max_depth=i, min_samples_leaf=j, random_state=0).fit(X_train_transformed, y_train)
            currentScore = mean_squared_error(y_test, clf.predict(X_test_transformed))
            print("MSE score pour i = ", i, " = ", currentScore)
            if currentScore< minScore:
                minScore = currentScore
                max_depth = i
                min_leaf = j

    print("\nRésultat trouvé : max depth = ", max_depth, " min leaf = ", min_leaf)
    print("\nAvec un score MSE = ", minScore)
    '''
    

if __name__=="__main__":
    regression()