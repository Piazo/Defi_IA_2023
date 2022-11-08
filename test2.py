import pandas as pd
import features
import csv

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from sklearn.compose import make_column_transformer

#TODO: Quand j'aurais un GPU on verra si je peux utliser gpu_id sur XGB


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

    model = xgb.XGBRFRegressor(n_estimators = 500, max_depth = 5)
    # model = lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=1400, num_leaves=40, learning_rate=0.1)
    model.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, model.predict(X_train), squared=False)
    test_score = mean_squared_error(y_test, model.predict(X_test), squared=False)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    print("on est bon")


testModel()