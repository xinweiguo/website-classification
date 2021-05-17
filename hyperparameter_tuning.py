import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import os, sys

import time

if __name__ == "__main__": 
    start = time.time()
    if len(sys.argv) == 1: 
        print("Please enter as follows: python hyperparameter_tuning.py <shalla|dmoz>")
        sys.exit()
    if sys.argv[1] == 'shalla': 
        which_list = 'shalla'
    elif sys.argv[1] == 'dmoz': 
        which_list = 'dmoz'
    else: 
        print("Please enter as follows: python hyperparameter_tuning.py <shalla|dmoz>")
        sys.exit()

    with open('models/'+which_list+'_data/'+which_list+'_full_x.npy', 'rb') as f: 
        full_input = np.load(f)
    with open('models/'+which_list+'_data/'+which_list+'_full_y.npy', 'rb') as f: 
        Y_train = np.load(f, allow_pickle=True)

    data, labels = full_input, Y_train

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)
    
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 8)
    # Fit the random search model
    rf_random.fit(full_input, Y_train)

    print('\n BEST PARAMETERS: \n')
    print(rf_random.best_params_)
    with open('models/'+which_list+'_data/'+which_list+'_best_parameters.txt', 'w') as f: 
        for key, value in rf_random.best_params_.items(): 
            f.write('%s=%s,\n' % (key, value))

    # grid search: not used but can be (replace the param_grid with the output of the above best_params_)
    # from sklearn.model_selection import GridSearchCV
    # # Create the parameter grid based on the results of random search 
    
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [80, 90, 100, 110],
    #     'max_features': [2, 3],
    #     'min_samples_leaf': [3, 4, 5],
    #     'min_samples_split': [8, 10, 12],
    #     'n_estimators': [100, 200, 300, 1000]
    # }
    # # Create a base model
    # rf = RandomForestClassifier()
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
    #                         cv = 5, n_jobs = -1, verbose = 2)
    # grid_search.fit(full_input, Y_train)
    # print(grid_search.best_params_)

    end = time.time()
    print('Total time: {}'.format(end - start))