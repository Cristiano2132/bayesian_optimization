from sklearn.model_selection import train_test_split
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from matplotlib import pyplot as plt
from skopt import gp_minimize
from typing import Optional
import numpy as np
import joblib
import json
import os


class Optimizer:

    def __init__(self, space: list, model: any, model_name: str, n_calls: int = 30, checkpoint: Optional[bool] = False, model_type: Optional[str] = 'regressor'):
        self.__model_name = model_name
        self.__space = space
        self.__model = model
        self.__counter = 0
        self.__n_calls = n_calls
        self.best_param = None
        self.accuracy = None
        self.best_model = None
        self.__checkpoint = checkpoint
        self.__model_type = model_type

    def __model_regressor_evaluate(self, mdl, X, y)-> float:
        Xtr,  Xtest, ytr, ytest = train_test_split(
            X, y, test_size=0.2, random_state=0)
        mdl.fit(Xtr, ytr.ravel())
        ypred = mdl.predict(Xtest)
        sum_squared_error = np.sum((ypred - ytest)**2)
        n = len(ytest)
        mse = sum_squared_error/n
        return mse

    def __model_classifier_evaluate(self, mdl, X, y)->int:
        Xtr,  Xtest, ytr, ytest = train_test_split(
            X, y, test_size=0.2, random_state=0)
        mdl.fit(Xtr, ytr.ravel())
        ypred = mdl.predict(Xtest)
        right_evaluations = np.sum(
            np.array(list(ypred)) == np.array(list(ytest)))
        return right_evaluations

    def __get_objective_function(self, X, y, k_folds:int=3) -> any:
        if self.__model_type == 'regressor':
            @use_named_args(self.__space)
            def objective(**params):
                self.__model.set_params(**params)
                return self.__model_regressor_evaluate(mdl=self.__model, X=X, y=y)
            return objective
        else:
            @use_named_args(self.__space)
            def objective(**params):
                self.__model.set_params(**params)
                return self.__model_classifier_evaluate(mdl=self.__model, X=X, y=y)
            return objective

    def find_optimal_params(self, X, y) -> None:
        print(f"Wait: Finding the best parameters .....")
        checkpoint_dir = 'checkpoint'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        obj = self.__get_objective_function(X=X, y=y)
        if self.__checkpoint:
            def checkpoint_onestep(res) -> None:
                x0 = res.x_iters   # List of input points
                y0 = res.func_vals  # Evaluation of input points
                print('Current iter: ', self.__counter,
                      ' - Score ', res.fun,
                      ' - Args: ', res.x)
                filename_checkpoint = f'{checkpoint_dir}/{self.__model_name}_checkpoint.pkl'
                joblib.dump((x0, y0), filename_checkpoint)
                filename_calls = f'{checkpoint_dir}/{self.__model_name}_ncalls.json'
                with open(filename_calls, 'w', encoding='utf-8') as f:
                    json.dump({'counter': self.__counter}, f,
                              ensure_ascii=False, indent=4)
                self.__counter += 1

            filename_checkpoint = f'{checkpoint_dir}/{self.__model_name}_checkpoint.pkl'
            if os.path.exists(filename_checkpoint):
                with open(f'{checkpoint_dir}/{self.__model_name}_ncalls.json', 'r') as f:
                    saved_info_dict = json.load(f)
                    self.__counter = int(saved_info_dict.get('counter'))
                x0, y0 = joblib.load(
                    f'{checkpoint_dir}/{self.__model_name}_checkpoint.pkl')

                res_gp = gp_minimize(func=obj,
                                     x0=x0,  # already examined values for x
                                     y0=y0,  # observed values for x0
                                     dimensions=self.__space,
                                     n_calls=max(self.__n_calls - \
                                                 self.__counter, 10),
                                     callback=[checkpoint_onestep],
                                     random_state=0, n_jobs=-1)
            else:
                res_gp = gp_minimize(func=obj,
                                     dimensions=self.__space,
                                     n_calls=max(self.__n_calls -
                                                 self.__counter, 10),
                                     callback=[checkpoint_onestep],
                                     random_state=0, n_jobs=-1)
        else:
            res_gp = gp_minimize(func=obj,
                                 dimensions=self.__space,
                                 n_calls=max(self.__n_calls -
                                             self.__counter, 10),
                                 random_state=0, n_jobs=-1)

        print("Otimization had done ...")
        self.best_params = dict(zip([s.name for s in self.__space], res_gp.x))
        self.accuracy = res_gp.fun
        self.best_model = self.__model.set_params(**self.best_params)
        print(
            f'Trainin acuracy: {self.accuracy}\nBest params: {self.best_params}')
        plot_convergence(res_gp)
        plt.tight_layout()
        plt.savefig(f"{self.__model_name}.png")
        plt.close()
