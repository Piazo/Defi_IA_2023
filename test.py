import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

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
from sklearn.ensemble import RandomForestRegressor


import features
import pandas as pd

def testModel():
    df = features.addOrderRequest(pd.read_csv("./data/allData.csv"))
    # On enlève la première colonne, et on enlève l'avatar et le request order pour l'instant
    # df = df.drop(columns=["avatar_id"])
    df = features.prepareDataframe(df)

    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"]
    df.drop(["price"], axis=1, inplace=True)

    df.to_csv("beforeEncoder.csv")

    # On encode les données non numériques avec OneHotEncoder
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())

    # On crée le jeu de tests et d'entraînement
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # On normalise les données
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)


    bestModel = RandomForestRegressor(max_depth=45, min_samples_leaf=1, random_state=0).fit(X_train_transformed, y_train)
    currentScore = mean_squared_error(y_test, bestModel.predict(X_test_transformed))
    print('random forest', currentScore)

    mod1 = SGDClassifier(random_state=0)
    mod2 = DecisionTreeClassifier(random_state=0)
    mod3 = KNeighborsClassifier(n_neighbors=2)

    mod4 = StackingClassifier([ ('SGD', mod1),
                                ('Tree', mod2),
                                ('KNN', mod3),
                                ('RandForest', bestModel)],
                                final_estimator=DecisionTreeClassifier())

    for mod in (mod1, mod2, mod3, mod4):
        mod.fit(X_train_transformed, y_train)
        print(mod.__class__.__name__, mod.score(X_train_transformed, y_train))

testModel()
'''
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y)

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)


# model = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=100)

# model.fit(xtr, ytr)
# print(model.score(xte, yte))

model = RandomForestClassifier(n_estimators=100, max_depth=100)

model.fit(xtr, ytr)
print(model.score(xte, yte))


model = AdaBoostClassifier(n_estimators=100)

model.fit(xtr, ytr)
print(model.score(xte, yte))




mod1 = SGDClassifier(random_state=0)
mod2 = DecisionTreeClassifier(random_state=0)
mod3 = KNeighborsClassifier(n_neighbors=2)

mod4 = StackingClassifier([ ('SGD', mod1),
                            ('Tree', mod2),
                            ('KNN', mod3)],
                            final_estimator=KNeighborsClassifier())

print(mod4.fit(xtr, ytr).score(xte, yte))
# mod4 = VotingClassifier([   ('SGD', mod1),
#                             ('Tree', mod2),
#                             ('KNN', mod3)],
#                             voting='hard')

# for mod in (mod1, mod2, mod3, mod4):
#     mod.fit(X_train, y_train)
#     print(mod.__class__.__name__, mod.score(X_test, y_test))
'''









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