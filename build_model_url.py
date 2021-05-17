import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pickle
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import os, sys 

if __name__ == "__main__": 
    os.chdir(sys.path[0])

    which_list = ''

    if len(sys.argv) == 1: 
        print("Please enter as follows: python build_model_url.py <shalla|dmoz>")
        sys.exit()
    if sys.argv[1] == 'shalla': 
        which_list = 'shalla'
    elif sys.argv[1] == 'dmoz': 
        which_list = 'dmoz'
    else: 
        print("Please enter as follows: python build_model_url.py <shalla|dmoz>")
        sys.exit()

    names=['Url','Category']
    df_file_path = 'data/'+which_list+'.csv' # make sure you run generate_ground_truths.py first to generate the df csv! 

    if which_list == 'shalla': 
        shalla_df = pd.read_csv(df_file_path, names=names, na_filter=False)

        shalla_category_counts = shalla_df.groupby('Category').count()

        train_fraction = 0.9

        train_count = 0

        temp_train = []
        temp_test = []

        train_test_labels = np.array([])

        for category_count in shalla_category_counts.Url: 
            train_count = round(train_fraction * category_count)
            
            temp_train = np.zeros(train_count)
            temp_test = np.zeros(category_count - train_count) + 1 
            
            train_test_labels = np.append(train_test_labels, temp_train)
            train_test_labels = np.append(train_test_labels, temp_test)
            
        shalla_df['train_test'] = train_test_labels.astype(int)
        shalla_train = shalla_df.loc[shalla_df.train_test == 0, ['Url', 'Category']].reset_index(drop=True)
        shalla_test = shalla_df.loc[shalla_df.train_test == 1, ['Url', 'Category']].reset_index(drop=True)


        X_train = shalla_train['Url']
        y_train = shalla_train['Category']
        X_test = shalla_test['Url']
        y_test = shalla_test['Category']

        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        text_clf = text_clf.fit(X_train, y_train)

        n_iter_search = 5
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
        gs_clf = RandomizedSearchCV(text_clf, parameters, n_iter = n_iter_search)
        gs_clf = gs_clf.fit(X_train, y_train)

        file_to_dump = open('url_classifier_'+which_list+'.pickle', "wb")
        pickle.dump(gs_clf, file_to_dump)


        

    else: # dmoz
        df_original=pd.read_csv(df_file_path, names=names, na_filter=False)

        df_original_no_duplicates = df_original.drop_duplicates(subset=['Url'], keep='first')
        df_original_no_duplicates.index = np.arange(1, len(df_original_no_duplicates) + 1)

        idxs_cats_start = ( # idxs to split categories for test data
            1, 
            50000, 
            520000,
            535300,
            650000,
            710000,
            764200,
            793080,
            839730,
            850000,
            955250,
            1013000,
            1143000,
            1293000,
            1492000
        )
            
        n_per_cat = 2000

        # Let's get new idxs: 
        #   - taking into account deleted duplicates and 
        #   - so that the new test set is as close as possible to the original:

        idxs_cats_start_new = []
            
        for idx in idxs_cats_start:
            url_cat_start = df_original.iloc[idx]['Url']
            idx_cat_start = df_original_no_duplicates.index[df_original_no_duplicates['Url'] == url_cat_start]
            assert len(idx_cat_start) == 1
            assert url_cat_start == df_original[idx: idx+1]['Url'].values[0]
            idxs_cats_start_new.append(idx_cat_start[0])

        # Create new train and test sets:

        df_correct = df_original_no_duplicates.copy()
        dt_correct = pd.concat([df_correct[idx:idx+n_per_cat] for idx in idxs_cats_start_new], axis=0)

        for idx in idxs_cats_start_new:
            df_correct.drop(range(idx+1, idx+n_per_cat+1), inplace=True)
            # +1 since dt_correct used slicing which starts from 0, but df_correct idxs in drop start from 1

        X_train = df_correct['Url']
        y_train = df_correct['Category']
        X_test = dt_correct['Url']
        y_test = dt_correct['Category']

        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        text_clf = text_clf.fit(X_train, y_train)

        n_iter_search = 5
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
        gs_clf = RandomizedSearchCV(text_clf, parameters, n_iter = n_iter_search)
        gs_clf = gs_clf.fit(X_train, y_train)

        file_to_dump = open('url_classifier_'+which_list+'.pickle', "wb")
        pickle.dump(gs_clf, file_to_dump)
