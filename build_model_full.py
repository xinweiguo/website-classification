import pyodbc
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import trange
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
import textstat
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
import re
import nltk 
import os, sys
import sqlite3
import pickle

def process_url(url): 
    final_url = url 
    
    o = urlparse(url)
    if not o.scheme: 
        final_url = 'http://' + url 
        
    return final_url

def count_uri(url): 
    url = process_url(url)
    o = urlparse(url)
    count = 0 
    for item in o: 
        if item: 
            count += 1

    return count 

def parse_url(url): 
    split_url = re.split('\W', url)
    split_url = [x for x in split_url if x] 
    
    output = " ".join(split_url)
    
    output = output.lower()
    

    output = word_tokenize(output)

        
    return " ".join(output)

# functions for generating keywords
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

if __name__ == "__main__": 
    os.chdir(sys.path[0])

    which_list = ''

    if len(sys.argv) == 1: 
        print("Please enter as follows: python build_model_full.py <shalla|dmoz>")
        sys.exit()
    if sys.argv[1] == 'shalla': 
        which_list = 'shalla'
    elif sys.argv[1] == 'dmoz': 
        which_list = 'dmoz'
    else: 
        print("Please enter as follows: python build_model_full.py <shalla|dmoz>")
        sys.exit()

    all_bodies = []

    conn = sqlite3.connect('databases/url_data.db')
    cursor = conn.cursor()

    query = "SELECT * FROM " + which_list 
    cursor.execute(query)

    for row in cursor.fetchall(): 
        all_bodies.append(row[2])

    # generate a vocabulary first 
    count_vect = CountVectorizer(lowercase=False)
    X_train_counts = count_vect.fit_transform(all_bodies)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(X_train_tf.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(count_vect.get_feature_names(), sorted_items, 500)

    i = 0 
    vocabulary = defaultdict(int)
    for item in keywords.keys(): 
        vocabulary[item] = i 
        i+=1 

    if not os.path.exists(os.getcwd()+'/models/'):
        os.makedirs(os.getcwd()+'/models/')

    vocabulary_path = 'models/'+which_list+'_data/'+which_list+'_vocabulary.pickle'
    file_to_dump = open(vocabulary_path, "wb")
    pickle.dump(vocabulary, file_to_dump)

    query = "SELECT * FROM " + which_list 
    cursor.execute(query)

    model_url = ''
    model_category = '' 
    model_body = '' 
    model_html_tags = ''

    temp_text = '' 

    results = cursor.fetchall()

    length = len(results)

    X_train = pd.DataFrame(columns = ['url_length', 'url_num_segments', 'url_uri_count', 
                                    'text_length', 'text_num_words', 'text_noun_proportion', 'text_verb_proportion', 'text_adj_proportion', 
                                    'text_SMOG_index', 'text_FK_reading_level', 'html_num_unique_tags', 'html_proportion_hyperlinks', 'html_number_js']) 
    url_length = np.zeros(length)
    url_num_segments = np.zeros(length)
    url_uri_count = np.zeros(length)
    text_length = np.zeros(length)
    text_num_words = np.zeros(length)
    text_noun_proportion = np.zeros(length)
    text_verb_proportion = np.zeros(length)
    text_adj_proportion = np.zeros(length)
    text_SMOG_index = np.zeros(length)
    text_FK_reading_level = np.zeros(length)
    html_num_unique_tags = np.zeros(length)
    html_proportion_hyperlinks = np.zeros(length)
    html_number_js = np.zeros(length)

    Y_train = np.zeros(length).astype('object')

    for i in trange(len(results)): 
        model_url = results[i][0]
        model_category = results[i][1]
        model_body = results[i][2]
        model_html_tags = results[i][3]

        temp_text = word_tokenize(model_body)
        pos_tagged_list = nltk.pos_tag(temp_text) 
        total_words = len(pos_tagged_list)

        noun_count = 0 
        verb_count = 0
        adj_count = 0
        other_count = 0

        for item in pos_tagged_list: 
            if item[1] in ['NN', 'NNS', 'NNP', 'NNPS']: 
                noun_count += 1
            elif item[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: 
                verb_count += 1
            elif item[1] in ['JJ', 'JJR', 'JJS']: 
                adj_count += 1
            else: 
                other_count += 1


        model_html_tags_list = model_html_tags.split()

        hyperlinks_count = 0
        scripts_count = 0
        total_tags = len(model_html_tags_list) 

        for tag in model_html_tags_list: 
            if tag == 'a': 
                hyperlinks_count += 1
            elif tag == 'script': 
                scripts_count += 1 
        
        if total_words == 0: 
            total_words = 1
            
        url_length[i] = float(len(model_url))
        url_num_segments[i] = float(len(parse_url(model_url).split()))
        url_uri_count[i] = float(count_uri(model_url))
        text_length[i] = float(len(model_body))
        text_num_words[i] = float(len(model_body.split()))
        text_noun_proportion[i] = float(noun_count / total_words)
        text_verb_proportion[i] = float(verb_count / total_words)
        text_adj_proportion[i] = float(adj_count / total_words)
        text_SMOG_index[i] = float(textstat.smog_index(model_body))
        text_FK_reading_level[i] = float(textstat.flesch_kincaid_grade(model_body))
        html_num_unique_tags[i] = float(len(set(model_html_tags.split())))
        html_proportion_hyperlinks[i] = float(hyperlinks_count / total_tags)
        html_number_js[i] = float(scripts_count)

        Y_train[i] = model_category
        
    X_train['url_length'] = url_length
    X_train['url_num_segments'] = url_num_segments
    X_train['url_uri_count'] = url_uri_count
    X_train['text_length'] = text_length
    X_train['text_num_words'] = text_num_words
    X_train['text_noun_proportion'] = text_noun_proportion
    X_train['text_verb_proportion'] = text_verb_proportion
    X_train['text_adj_proportion'] = text_adj_proportion
    X_train['text_SMOG_index'] = text_SMOG_index
    X_train['text_FK_reading_level'] = text_FK_reading_level
    X_train['html_num_unique_tags'] = html_num_unique_tags
    X_train['html_proportion_hyperlinks'] = html_proportion_hyperlinks
    X_train['html_number_js'] = html_number_js

    final_cv = CountVectorizer(analyzer='word', lowercase=False, vocabulary=vocabulary)
    X_to_add = final_cv.fit_transform(all_bodies)
        
    full_input = np.hstack((X_train.values.astype(float), X_to_add.toarray()))
    
    # save the raw data array, so you don't need to redo this process each time 

    if not os.path.exists(os.getcwd()+'/models/'+which_list+'_data/'):
        os.makedirs(os.getcwd()+'/models/'+which_list+'_data/')

    with open('models/'+which_list+'_data/'+which_list+'_full_x.npy', 'wb') as f: 
        np.save(f, full_input)
    with open('models/'+which_list+'_data/'+which_list+'_full_y.npy', 'wb') as f: 
        np.save(f, Y_train)

    # the parameters here are chosen as the result of hyperparameter_tuning.py through performing a randomized search over a paramter grid 
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(full_input, Y_train)

    model_path = 'models/'+which_list+'_full.pickle'
    file_to_dump = open(model_path, "wb")
    pickle.dump(clf, file_to_dump)