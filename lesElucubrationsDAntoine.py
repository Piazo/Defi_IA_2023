import pandas as pd
import features
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RANSACRegressor
from sklearn.svm import LinearSVR
import lightgbm as lgb

from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error

import csv
import matplotlib.pyplot as plt
import numpy as np

def testModel(pred = False):
    # On récupère le dataFrame et on prepare tout le bordel
    # df = features.prepareDataframe(features.addOrderRequest(pd.read_csv("./data/allData.csv")))
    # df.to_csv("ceciestuntest.csv")
    df = pd.read_csv('ceciestuntest.csv')
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"].astype("float")
    df.drop(["price"], axis=1, inplace=True)

    # print(df.dtypes, y.dtypes)
    
    """
        encoder = OneHotEncoder()
        X = encoder.fit_transform(df.values)
    """

    # Essayer d'encoder la col hotel_id
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())
    # print(df.dtypes)
    df.to_csv('afterEncod.csv')

    # for category in encoder.categories_:
    #     print(category[:5])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

    # On standardise les données
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Feature data dimension: ", X_train.shape)

    # select = SelectKBest(score_func=f_regression, k=8)
    # z = select.fit_transform(X_train, y_train) 
    # filter = select.get_support()
    # print(filter)
    # features = np.array(df.columns)
    # print("All features:")
    # print(features)

    # print("Selected best 8:")
    # print(features[filter])
    # print(z) 

    # svm = LinearSVR()
    # svm.fit(X_train, y_train)

    # model = RANSACRegressor()
    # model.fit(X_train, y_train)

    # model = SGDRegressor(penalty='elasticnet')
    # model.fit(X_train, y_train)

    #Meilleur resultat obtenu avec n_estimator = 10000 et num_leaves=40
    # model = lgb.LGBMRegressor(n_estimators=1000, num_leaves=30)
    # model.fit(X_train, y_train)

    model = GradientBoostingRegressor(n_estimators = 1000, max_depth=5)
    model.fit(X_train, y_train)
    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, model.predict(X_train))
    test_score = mean_squared_error(y_test, model.predict(X_test))
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)

    # N, train_score2, val_score = learning_curve(model, X_train, y_train, cv=4, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1,1,10))

    # plt.figure(figsize=(12,8))
    # plt.plot(N, train_score2.mean(axis=1))
    # plt.plot(N, val_score.mean(axis=1))
    # plt.show()

    # lgb.plot_importance(gbr, max_num_features=10)
    # plt.show()

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