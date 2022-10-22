import pandas as pd
import features
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor

from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error

import csv

def testModel(pred = False):
    # On récupère le dataFrame et on prepare tout le bordel
    # df = features.prepareDataframe(features.addOrderRequest(pd.read_csv("./data/allData.csv")))
    # df.to_csv("ceciestuntest.csv")
    df = pd.read_csv('ceciestuntest.csv')
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"]
    df.drop(["price"], axis=1, inplace=True)

    # print(df.dtypes, y.dtypes)
    
    """
        encoder = OneHotEncoder()
        X = encoder.fit_transform(df.values)
    """
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())

    # for category in encoder.categories_:
    #     print(category[:5])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    # On standardise les données
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # svm = LinearSVR()
    # svm.fit(X_train, y_train)

    gbr = LGBMRegressor(n_estimators=50000, num_leaves=20)
    gbr.fit(X_train, y_train)

    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, gbr.predict(X_train))
    test_score = mean_squared_error(y_test, gbr.predict(X_test))
    
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
        print(test_data.columns)
        
        # On normalise les données en se basant sur le training set
        X_test_data_transformed = scaler.transform(test_data)

        # On génère le csv
        header = ["index", "price"]
        data = []
        for i in range(len(X_test_data_transformed)):
            prediction = [i, int(gbr.predict([X_test_data_transformed[i]]))]
            data.append(prediction)

        with open('predictionsKaggle.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write data
            writer.writerows(data)


testModel(True)