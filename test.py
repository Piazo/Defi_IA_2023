import imp
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import features
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from numba import jit, cuda
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



def testModel():
# On récupère le dataFrame
    df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))
    
    # # On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
    # df = df.drop(columns=["avatar_id"])

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

    # On standardise les données
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # mod1 = RandomForestRegressor(max_depth=45, min_samples_leaf=1, random_state=0)

    # mod2 = MLPRegressor(hidden_layer_sizes=(16,16,16,16,16,1), random_state=0, max_iter=3000)
    # for i in range(1,21,5):
    #     for j in range(1,21,5):
    #         for k in range(1,21,5):
    #             mod2 = MLPRegressor(hidden_layer_sizes=(i,j,k), random_state=0, max_iter=3000)
    #             mod2.fit(X_train_transformed, y_train)
    #             score = mean_squared_error(y_test, mod2.predict(X_test_transformed))
    #             print(i,j,k, score)


    # mod3 = KNeighborsClassifier(n_neighbors=2)

    # mod4 = VotingRegressor([    ('Rfor', mod1),
    #                             ('mlp', mod2),],
    #                             n_jobs=-1)

    # print(mod4.fit(X_train_transformed, y_train).score(X_test_transformed, y_test))

    # for mod in (mod1, mod2, mod4):
    #     mod.fit(X_train_transformed, y_train)
    #     score = mean_squared_error(y_test, mod.predict(X_test_transformed))
    #     print(mod.__class__.__name__, mod.score(X_test_transformed, y_test), " score : ", score)

    # currentScore = mean_squared_error(y_test, mod4.predict(X_test_transformed))

    # crossval = KFold(5)
    currentScore = cross_val_score(RandomForestRegressor(max_depth=45, min_samples_leaf=1), X_train_transformed, y_train, cv=5)

    # bestModel = RandomForestRegressor(max_depth=45, min_samples_leaf=1, random_state=0).fit(X_train_transformed, y_train)
    # currentScore = mean_squared_error(y_test, bestModel.predict(X_test_transformed))
    print(currentScore)

testModel()


    # mod4 = VotingClassifier([   ('SGD', mod1),
    #                             ('Tree', mod2),
    #                             ('KNN', mod3)],
    #                             voting='hard')



"""
---------------- Note pour Antonio ----------------

Utiliser des ensembles de modeles

On va utiliser du BAGGING lorsque qu'on a plusieurs modeles qui vont avoir tendence a faire de l'OVERFITTING

On va utiliser du BOOSTING lorsque qu'on a plusieurs modeles qui vont avoir tendence a faire de l'UNDERFITTING

On va utiliser du STACKING lorsque qu'on a plusieurs modeles que l'on va avoir entrainé avec beaucoup de données




Pour la cross validation on peut utiliser validation_curve()

Trouver les meilleurs param avec GridSearchCV

Regarder la confuision_matrix

Pour voir si il est utile de continuer a augmenter la taille des données : learning_curve()
"""