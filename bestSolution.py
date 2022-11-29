from tabnanny import verbose
import pandas as pd
import features
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error

import csv


#TODO: Essayer de regrouper les labels hotel et noms d'hotel et changer les hotel id en label plutot que en int
#TODO: checker import optuna pour optimiser les hyperparametres

def testModel(pred = False):

####################### Preparation des dataframes #######################
    # On récupère le dataFrame et on prepare tout le bordel
    df = features.prepareDataframe(pd.read_csv("./data/allData.csv"))

    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"]
    df.drop(["price", "Unnamed: 0"], axis=1, inplace=True)
###########################################################################


####################### Encodage et standardisation des donnees #######################
    # Essayer d'encoder la col hotel_id
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    # On standardise les données
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
########################################################################################

################### GOAT PREDICTOR PR LE MOMENT ###################
    # Meilleur resultat obtenu avec n_estimator = 50000 et num_leaves=40
    print("starting regression...")
    model = lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=3000, num_leaves=40, learning_rate=0.1)
    # model = xgb.XGBRegressor(n_estimators = 10000, max_depth = 7)

###################################################################

    model.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, model.predict(X_train), squared=False)
    test_score = mean_squared_error(y_test, model.predict(X_test), squared=False)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)


    if(pred == True):
        # On traite les données de test_set.csv
        test_data = pd.read_csv("./data/test_set.csv")
        test_data = test_data.drop(columns=["index"])
        # On ajoute les caractéristiques des hôtels
        test_data = features.prepareDataframe(test_data)
        # On encode les données non numériques avec OneHotEncoder
        columns_transfo = make_column_transformer(
            (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
            remainder='passthrough')
        transformed = columns_transfo.fit_transform(test_data).toarray()
        test_data = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())

        test_data = features.rearrangeCol(df, test_data)
        # print(test_data.columns)
        
        # On normalise les données en se basant sur le training set
        X_test_data_transformed = scaler.transform(test_data)

        # On génère le csv
        header = ["index", "price"]
        data = []
        for i in range(len(X_test_data_transformed)):
            prediction = [i, int(model.predict([X_test_data_transformed[i]]))]
            data.append(prediction)

        with open('predictionsKaggle.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write data
            writer.writerows(data)


testModel(True)