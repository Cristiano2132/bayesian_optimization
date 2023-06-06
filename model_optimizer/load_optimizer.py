from interfaces.optimizer import Optimizer
from model_optimizer.models.lasso import LassoOptimizer
from model_optimizer.models.rf import RandomForestOptimizer
from typing import Type


class LoadOptimizer():
    def __init__(self, regressor_name: str, X, y, checkpoint_prefix: str, k_folds: int = 3)->None:
        self.__regressor_name = regressor_name
        self.__checkpoint_prefix = checkpoint_prefix 
        self.__X = X
        self.__y = y
        self.__k_folds = k_folds

    def create_regressor(self)-> any:
        print(f'Regressor name: {self.__regressor_name}')
        return self.__select_regressor(regressor_name=self.__regressor_name).get_regression()


    def __load_rf_regressor(self)-> any:
        regressor = RandomForestOptimizer(X=self.__X, y=self.__y, checkpoint_prefix=self.__checkpoint_prefix, k_folds=self.__k_folds)
        return regressor


    def __load_lasso_regressor(self)-> any:
        regressor = LassoOptimizer(X=self.__X, y=self.__y, checkpoint_prefix=self.__checkpoint_prefix, k_folds=self.__k_folds)
        return regressor


    def __select_regressor(self, regressor_name: str)-> Type[Optimizer]:
        switcher = {
            'lasso': self.__load_lasso_regressor,
            'rf': self.__load_rf_regressor,
        }
        regressor = switcher.get(regressor_name)
        return regressor()
