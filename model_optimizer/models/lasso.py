import json
import os
import time
import numpy as np
from interfaces.optimizer import Optimizer
from sklearn.model_selection import cross_val_score
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.linear_model import Lasso
from skopt.plots import plot_convergence
from matplotlib import pyplot as plt
import pandas as pd
import joblib



class LassoOptimizer(Optimizer):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, checkpoint_prefix: str, k_folds: int = 3):
        self.__X = X
        self.__y = y.values.ravel()
        self.__checkpoint_prefix = checkpoint_prefix
        self.__counter = 1
        self.__timer = 0
        self.__n_calls = 20
        self.__k_folds = k_folds

    def __get_space(self) -> list:
        space = [Real(0, 0.02, name='alpha')]
        return space

    def __get_params(self) -> list:
        space_params = ['alpha']
        return space_params

    def __get_objective_function(self) -> any:
        reg = Lasso(random_state=0, max_iter=10000, normalize=True)
        space = self.__get_space()

        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg, self.__X, self.__y, cv=self.__k_folds, n_jobs=-1,
                                            scoring="neg_root_mean_squared_error"))
        return objective

    def __find_optimal_params(self) -> dict:

        print(f"Wait: Finding the best parameters .....")
        obj = self.__get_objective_function()
        space = self.__get_space()
        space_params = self.__get_params()
        is_checkpoint = os.path.isfile( f'{self.__checkpoint_prefix}_checkpoint.pkl')
        def onstep(res):
            
            x0 = res.x_iters   # List of input points
            y0 = res.func_vals # Evaluation of input points
            print('Last eval: ', x0[-1], 
                ' - Score ', y0[-1])
            print('Current iter: ', self.__counter, 
                ' - Score ', res.fun, 
                ' - Args: ', res.x,
                'time', self.__timer )
            joblib.dump((x0, y0), f'{self.__checkpoint_prefix}_checkpoint.pkl') # Saving a checkpoint to disk

            with open(f'{self.__checkpoint_prefix}_ncalls.json', 'w', encoding='utf-8') as f:
                json.dump({'counter': self.__counter, 'timer': self.__timer}, f, ensure_ascii=False, indent=4)
            time_passed = time.time() - start_time
            self.__timer += time_passed
            self.__counter += 1

        start_time = time.time()
        if is_checkpoint:
            with open(f'{self.__checkpoint_prefix}_ncalls.json', 'r') as f:
                saved_info_dict = json.load(f)
                self.__counter = int(saved_info_dict.get('counter'))
                self.__timer = int(saved_info_dict.get('timer'))
            x0, y0 = joblib.load(f'{self.__checkpoint_prefix}_checkpoint.pkl')
            res_gp = gp_minimize(func=obj, 
                                x0=x0, # already examined values for x
                                y0=y0,  # observed values for x0
                                dimensions=space, 
                                n_calls=max(self.__n_calls - self.__counter, 10), 
                                callback=[onstep], 
                                random_state=0, n_jobs=-1) 

        else:
            res_gp = gp_minimize(func=obj, dimensions=space, n_jobs=-1,
                                n_calls=self.__n_calls, callback=[onstep], random_state=0)
        time_passed = time.time() - start_time
        self.__timer += time_passed

        print("Otimization had done ...")
        best_params = dict(zip(space_params, res_gp.x))
        dict_results = {}
        dict_results['rf'] = {
            "Accuracy": res_gp.fun,
            "Best params": best_params
        }
        print(f'Trainin acuracy: {res_gp.fun}\nBest params: {best_params}')
        plot_convergence(res_gp)
        plt.tight_layout()
        plt.savefig(f"{self.__checkpoint_prefix}_{len(self.__X)}.png")
        return dict_results


    def get_regression(self):

        dict_results = self.__find_optimal_params()
        best_params = dict_results.get('Lasso').get('Best params')
        model = Lasso(alpha=best_params.get('alpha'),
                      random_state=0, max_iter=10000, normalize=True)
        return model, best_params
