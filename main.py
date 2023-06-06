from bayesian_opt import Optimizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from skopt.space import Integer, Real
import pandas as pd
import numpy as np
from scipy.stats import skew


def print_missing(X: pd.DataFrame) -> None:
    missing_rate = (X.isnull().sum() / len(X)) * 100
    missing_rate = missing_rate.drop(
        missing_rate[missing_rate == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': missing_rate})
    print(missing_data.head(20))


if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    df_train.drop("Id", axis=1, inplace=True)
    df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
    numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index
    skewed_feats = df_train[numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    df_train[skewed_feats] = np.log1p(df_train[skewed_feats])
    df_train = pd.get_dummies(df_train)
    df_train = df_train.fillna(df_train.mean())
    print_missing(df_train)

    y = df_train.SalePrice.values.astype(float)
    X = df_train.reset_index(drop=True).drop(['SalePrice'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    dict_models = {
        'lasso': {
            'model': Lasso(),
            'space': [Real(0, 0.02, name='alpha')]},

        'rf': {
            'space': [Integer(100, 1000, name='n_estimators'),
                      Integer(2, 100, name='min_samples_split'),
                      Integer(1, 10, name='min_samples_leaf')
                      ],
            'model': RandomForestRegressor()}
    }

    for model in dict_models:
        model_name = model
        space = dict_models[model]['space']
        model = dict_models[model]['model']
        optimizer = Optimizer(space=space, model=model,
                              model_name=model_name, n_calls=30, checkpoint=True)

        optimizer.find_optimal_params(X=X_train, y=y_train)
        best_model = optimizer.best_model.fit(X_train, y_train.ravel())
        y_pred = best_model.predict(X_test)
        print(
            f"Test accuracy: cor: {np.corrcoef(y_pred, y_test)[0,1]:.2f}, mse: {np.mean((y_pred - y_test)**2):.2f}")
