#import lightgbm
import pandas as pd
from numpy.core.fromnumeric import var
from numpy.random.mtrand import sample
from scipy.sparse.construct import rand
from sklearn import svm
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.datasets import fetch_openml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.naive_bayes import GaussianNB
#from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import (
    LogisticRegression,
    BayesianRidge,
    Lars,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#from pycaret.datasets import get_data
from sklearn.tree import DecisionTreeClassifier
import random

class BlackBoxSimulator:
    """Simulates black box models for specified dataset"""

    def load_breast_cancer_utilities(debug=True):
        """[summary]

        Args:
            debug (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        # Loading the dataset using sklearn
        data = load_breast_cancer()

        # Separating data into feature variable X and target variable y respectively
        X = data["data"]
        y = data["target"]

        # Extracting the names of the features from data
        features = data["feature_names"]

        # Storing the indices of discrete valued features
        discrete_features = []

        # Splitting X & y into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.90, random_state=50
        )
        
        clf = MLPClassifier(random_state=50).fit(X_train, y_train)
        # clf = DecisionTreeClassifier(random_state=50)
        clf.fit(X_train, y_train)
        
        np.random.seed(50)
        sample_idx = np.random.permutation(X_test.shape[0])[:10]
        # Checking the model's performance on the test set
        if debug:
            print("R2 score for the model on test set =", clf.score(X_test, y_test))

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "features": features,
            "model": clf,
            "mode": "classification",
            "discrete_features": discrete_features,
            "sample_idx": sample_idx,
        }
