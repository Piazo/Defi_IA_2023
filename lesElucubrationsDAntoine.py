import pandas as pd
import features
from numba import jit, cuda
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RANSACRegressor, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.svm import LinearSVR
import lightgbm as lgb
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 


def testModel(pred = False):

####################### Preparation des dataframes #######################
    # On récupère le dataFrame et on prepare tout le bordel
    # df = features.prepareDataframe(features.addOrderRequest(pd.read_csv("./data/allData.csv")))
    # df.to_csv("ceciestuntest.csv")

    df = pd.read_csv('ceciestuntest.csv')
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    # on récupère la colonne cible, le prix, et on la supprime
    y = df["price"]
    df.drop(["price"], axis=1, inplace=True)
###########################################################################


####################### Encodage et standardisation des donnees #######################
    # Essayer d'encoder la col hotel_id
    columns_transfo = make_column_transformer(
        (OneHotEncoder(), ['brand', 'group', 'city', 'language']), 
        remainder='passthrough')
    transformed = columns_transfo.fit_transform(df).toarray()
    df = pd.DataFrame(transformed, columns=columns_transfo.get_feature_names_out())
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.95, random_state=0)

    # On standardise les données
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
########################################################################################

####################### Cross validation part #######################
    n_folds = 5
    def rmsle_cv(model):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
        rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)
#####################################################################
    """
    TabModel = []

    print("Starting to process models...")

    enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3, max_iter=3000))
    # TabModel.append(enet)

    gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    # TabModel.append(gboost)

    krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    # TabModel.append(krr)

    svr = LinearSVR()
    TabModel.append(svr)

    ransac = RANSACRegressor()
    TabModel.append(ransac)

    sgdr = SGDRegressor(penalty='elasticnet')
    TabModel.append(sgdr)

    xgbr = xgb.XGBRegressor(max_depth=3, n_estimators=2200)
    TabModel.append(xgbr)


    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    # TabModel.append(lasso)

    stacked_averaged_models = StackingAveragedModels(base_models = (enet, gboost, krr),
                                                    meta_model = lasso)
    # TabModel.append(stacked_averaged_models)

    if len(TabModel) > 0:
        for mod in TabModel:
            print("1er model : ", mod)
            print(mod.__class__.__name__, " cross validation score : ", rmsle_cv(mod).mean())
            mod.fit(X_train, y_train)
            print(mod.__class__.__name__, mod.score(X_test, y_test), " score : ", mean_squared_error(y_test, mod.predict(X_test)))
    """

################### GOAT PREDICTOR PR LE MOMENT ###################
    # Meilleur resultat obtenu avec n_estimator = 10000 et num_leaves=40
    # model = lgb.LGBMRegressor(n_estimators=100000, num_leaves=40)
    # model.fit(X_train, y_train)
###################################################################

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


    

    #Meilleur resultat obtenu avec n_estimator = 10000 et num_leaves=40
    # model = lgb.LGBMRegressor(num_leaves=40, n_estimators=10000)

    # model = GradientBoostingRegressor(n_estimators = 1000, max_depth=5)

    # model = RandomForestRegressor()

    model = RANSACRegressor()
    model.fit(X_train, y_train)

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