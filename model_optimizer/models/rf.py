from typing import Optional
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
import pandas as pd
from skopt.plots import plot_convergence
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
import joblib
import time
import json
import os
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


class Optimizer:

    def __init__(self, space: list, model: any, model_name: str, n_calls: int = 30):
        self.__model_name = model_name
        self.__space = space
        self.__model = model
        self.__counter = 0
        self.__n_calls = n_calls
        self.best_param = None
        self.accuracy = None
        self.best_model = None

    def __model_evaluate(self, mdl, X, y) -> int:
        Xtr,  Xtest, ytr, ytest = train_test_split(
            X, y, test_size=0.2, random_state=0)
        mdl.fit(Xtr, ytr.ravel())
        ypred = mdl.predict(Xtest)
        sum_squared_error = np.sum((ypred - ytest)**2)
        n = len(ytest)
        mse = sum_squared_error/n
        return mse

    def __get_objective_function(self, X, y, k_folds=3) -> any:
        @use_named_args(self.__space)
        def objective(**params):
            self.__model.set_params(**params)
            return self.__model_evaluate(mdl=self.__model, X=X, y=y)
        return objective

    def find_optimal_params(self, X, y) -> dict:
        print(f"Wait: Finding the best parameters .....")
        obj = self.__get_objective_function(X=X, y=y)

        def checkpoint_steps_monitoring(res) -> None:
            x0 = res.x_iters   # List of input points
            y0 = res.func_vals  # Evaluation of input points
            # print('Last eval: ', x0[-1],
            #     ' - Score ', y0[-1])
            print('Current iter: ', self.__counter,
                  ' - Score ', res.fun,
                  ' - Args: ', res.x)
            filename_checkpoint = f'{self.__model_name}_checkpoint.pkl'
            # Saving a checkpoint to disk
            joblib.dump((x0, y0), filename_checkpoint)
            filename_calls = f'{self.__model_name}_ncalls.json'
            with open(filename_calls, 'w', encoding='utf-8') as f:
                json.dump({'counter': self.__counter}, f,
                          ensure_ascii=False, indent=4)
            self.__counter += 1

        filename_checkpoint = f'{self.__model_name}_checkpoint.pkl'
        if os.path.exists(filename_checkpoint):
            with open(f'{self.__model_name}_ncalls.json', 'r') as f:
                saved_info_dict = json.load(f)
                self.__counter = int(saved_info_dict.get('counter'))
            x0, y0 = joblib.load(f'{self.__model_name}_checkpoint.pkl')

            res_gp = gp_minimize(func=obj,
                                 x0=x0,  # already examined values for x
                                 y0=y0,  # observed values for x0
                                 dimensions=space,
                                 n_calls=max(self.__n_calls - \
                                             self.__counter, 10),
                                 callback=[checkpoint_steps_monitoring],
                                 random_state=0, n_jobs=-1)
        else:
            res_gp = gp_minimize(func=obj,
                                 dimensions=space,
                                 n_calls=max(self.__n_calls -
                                             self.__counter, 10),
                                 callback=[checkpoint_steps_monitoring],
                                 random_state=0, n_jobs=-1)

        print("Otimization had done ...")
        self.best_params = dict(zip([s.name for s in self.__space], res_gp.x))
        self.accuracy = res_gp.fun
        self.best_model = self.__model.set_params(
            **{'n_jobs': -1, **self.best_params})
        print(
            f'Trainin acuracy: {self.accuracy}\nBest params: {self.best_params}')
        plot_convergence(res_gp)
        plt.tight_layout()
        plt.savefig(f"{self.__model_name}_{len(X)}.png")


def print_missing(X: pd.DataFrame) -> None:
    missing_rate = (X.isnull().sum() / len(X)) * 100
    missing_rate = missing_rate.drop(
        missing_rate[missing_rate == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': missing_rate})
    print(missing_data.head(20))


if __name__ == '__main__':

    df_train = pd.read_csv('data/train.csv')
    print(df_train.shape)
    # df_test = pd.read_csv('data/test.csv')
    df_train.drop("Id", axis=1, inplace=True)
    # df_test.drop("Id", axis = 1, inplace = True)

    # print("all_data size is : {}".format(df_train.shape))
    # # print_missing(X)
    # var_null_none = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 'MSSubClass',
    #                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType']
    # var_null_0 = ["MasVnrArea"]
    # values = {"Functional": "Typ", **
    #           {v: 'None' for v in var_null_none}, **{i: 0 for i in var_null_0}}
    # df_train.fillna(value=values, inplace=True)
    # df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(
    #     lambda x: x.fillna(x.median()))
    # df_train = df_train.drop(['Utilities'], axis=1)
    # # MSSubClass=The building class
    # df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)

    # # Changing OverallCond into a categorical variable
    # df_train['OverallCond'] = df_train['OverallCond'].astype(str)

    # # Year and month sold are transformed into categorical features.
    # df_train['YrSold'] = df_train['YrSold'].astype(str)
    # df_train['MoSold'] = df_train['MoSold'].astype(str)
    # from sklearn.preprocessing import LabelEncoder
    # cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    #         'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
    #         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
    #         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
    #         'YrSold', 'MoSold')
    # # process columns, apply LabelEncoder to categorical features
    # for c in cols:
    #     lbl = LabelEncoder()
    #     lbl.fit(list(df_train[c].values))
    #     df_train[c] = lbl.transform(list(df_train[c].values))
    # df_train.dropna(inplace=True)
    # log transform the target:
    df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

    skewed_feats = df_train[numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    df_train[skewed_feats] = np.log1p(df_train[skewed_feats])
    df_train = pd.get_dummies(df_train)
    # filling NA's with the mean of the column:
    df_train = df_train.fillna(df_train.mean())
    # print_missing(X)
    y = df_train.SalePrice.values.astype(float)
    X = df_train.reset_index(drop=True).drop(['SalePrice'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    space = [Integer(100, 1000, name='n_estimators'),
             Integer(2, 100, name='min_samples_split'),
             Integer(1, 10, name='min_samples_leaf')
             ]
    model = RandomForestRegressor()
    model_name = 'rf'
    optimizer = Optimizer(space=space, model=model,
                          model_name=model_name, n_calls=30)

    optimizer.find_optimal_params(X=X_train, y=y_train)
    best_model = optimizer.best_model.fit(X_train, y_train.ravel())
    y_pred = best_model.predict(X_test)
    print(
        f"Test accuracy: cor: {np.corrcoef(y_pred, y_test)[0,1]:.2f}, mse: {np.mean((y_pred - y_test)**2):.2f}")
