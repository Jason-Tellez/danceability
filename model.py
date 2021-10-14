#################
#### Imports ####
#################

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


from sklearn.metrics import median_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.neighbors import KNeighborsRegressor




#################
##### Model #####
#################

def X_y_split(train, validate, test, target):
    """
    Functions that takes in trainm validate, test, and target var and split to X and y datasets
    """
    # Setup X and y
    X_train = train.drop(columns=target)
    y_train = train[target]

    X_validate = validate.drop(columns=target)
    y_validate = validate[target]

    X_test = test.drop(columns=target)
    y_test = test[target]
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def regression_errors(df, y, yhat):
    """
    Function creates variables holding the metrics for a given model's predictions
    
    Returns SSE, ESS, TSS, MSE, RMSE
    """
    residual = df[yhat] - df[y]
    residual_sq = residual**2
    SSE = residual_sq.sum()
    ESS = ((df[yhat] - df[y].mean())**2).sum()
    TSS = SSE + ESS
    MSE = SSE/len(df[y])
    RMSE = MSE**0.5
    return SSE, ESS, TSS, MSE, RMSE


def select_feats(scaled_df, k, target):
    """
    Function uses Select K Best and Recursive Feature Elimination to return two lists of the best features chosen by each method
    """
    # kbest
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(scaled_df, target)
    X_kbest = scaled_df.columns[kbest.get_support()]

    # recursive feature elimination
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(scaled_df, target)
    X_rfe = scaled_df.columns[rfe.get_support()]
    return X_kbest, X_rfe