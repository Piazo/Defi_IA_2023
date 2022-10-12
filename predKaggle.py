import model
import features
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor

def createCSV():

    # On récupère le modèle
    # mdl = model.regression()

    # On traite les données de test_set.csv
    test_data = pd.read_csv("./data/test_set.csv")

    #On supprime les colonnes index, avatar et order_request pour l'instant
    test_data = test_data.drop(columns=["index", "avatar_id", "order_requests"])

    # On ajoute les caractéristiques des hôtels
    test_data = features.prepareDataframe(test_data)

    print(test_data)

    # On encode les données non numériques avec OneHotEncoder
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(test_data).toarray()
    test_data = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())
    
    print(test_data)

    # On normalise les données
    scaler = StandardScaler().fit(test_data)
    X_test_transformed = scaler.transform(test_data)


    '''
    header = ["ID", "Category"]
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


if __name__=="__main__":
    createCSV()