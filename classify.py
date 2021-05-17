import pickle 
import os, sys
import csv 
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
import nltk 
from nltk.tokenize import word_tokenize
from tqdm import trange
from word_segmentation import ngrams 
from string import digits
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import textstat 

def process_url(url): 
    final_url = url 
    
    o = urlparse(url)
    if not o.scheme: 
        final_url = 'http://' + url 
        
    return final_url

def remove_scheme(url): 
    o = urlparse(url) 
    
    start_index = 0
    if o.scheme: 
        start_index = len(o.scheme)
    while start_index < len(url) and url[start_index].isalnum() != True: 
        start_index += 1
    
    return url[start_index:]

def read_input(where_str): 
    input_list = []
    if where_str == 'file': 
        with open(sys.argv[5], 'r') as f: 
            temp_list = csv.reader(f, delimiter=',')
            for row in temp_list: 
                input_list += [item.lstrip().rstrip() for item in list(row) if item.lstrip().rstrip()]

    else: 
        temp_list = sys.argv[5:] 
        input_list = [item.lstrip().rstrip() for item in temp_list]

    return input_list 


def write_output(where_str, url_list, contents): 
    """
    contents is a list-like of the predicted categories of url_list items
    """
    output_str = '' 

    for i in range(len(url_list)): 
        output_str += '{}: {}'.format(url_list[i], contents[i])
        if i != len(url_list) - 1: 
            output_str += '\n'

    if where_str == 'stdout':
        print(output_str)

    else: # print to file 
        with open('classifier_output.txt', 'w') as f: 
            f.write(output_str) 
     
# below is just for full input 
def parse_body(raw_body_text): 
    split_body = re.split('\W', raw_body_text)
    split_body = [x for x in split_body if x] 
    
    output = " ".join(split_body)
    
    output = output.lower()
    

    output = word_tokenize(output)

    return " ".join(output) 

def parse_url(url): 
    split_url = re.split('\W', url)
    split_url = [x for x in split_url if x] 
    
    output = " ".join(split_url)
    
    output = output.lower()

    output = word_tokenize(output)

    return " ".join(output)

def fetch_everything(url): 
    full_url = process_url(url)

    res = requests.get(full_url, timeout = 5)

    body_str = ''
    soup = BeautifulSoup(res.text, "html.parser")

    tag = soup.body
    
    try: 
        for string in tag.strings:
            body_str += string
    except: 
        body_str = '' 

    return (parse_body(body_str), res.text)

def count_uri(url): 
    url = process_url(url)
    o = urlparse(url)
    count = 0 
    for item in o: 
        if item: 
            count += 1

    return count 

def generate_prediction(url_list, model): 
    predictions = [] 

    for url in url_list: 
        model_url = url
        temp_stuff = fetch_everything(process_url(url))
        model_body = temp_stuff[0]

        if model_body == '': 
            predictions.append('UNABLE TO FETCH WEBSITE BODY')
            continue 
        
        model_body = [item.lstrip(digits).rstrip(digits) for item in model_body.split()]
        output = [] 

        for item in model_body: 
            for item2 in ngrams.segment2(item)[1]: 
                if item2: 
                    output.append(item2) 
        model_body = ' '.join(output)
  
        temp_raw_html = temp_stuff[1]

        if temp_raw_html == '': 
            predictions.append('UNABLE TO FETCH WEBSITE HTML')
            continue 

        soup = BeautifulSoup(temp_raw_html, "html.parser")

        temp_all_tags = " ".join([tag.name for tag in soup.find_all()])
        model_html_tags = temp_all_tags

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
        temp_df_contents = {'url_length': len(model_url), 
                    'url_num_segments': len(parse_url(model_url).split()), 
                    'url_uri_count': count_uri(model_url),       
                    'text_length': len(model_body), 
                    'text_num_words': len(model_body.split()), 
                    'text_noun_proportion': noun_count / total_words, 
                    'text_verb_proportion': verb_count / total_words, 
                    'text_adj_proportion': adj_count / total_words, 
                    'text_SMOG_index': textstat.smog_index(model_body), 
                    'text_FK_reading_level': textstat.flesch_kincaid_grade(model_body), 
                    'html_num_unique_tags': len(set(model_html_tags.split())), 
                    'html_proportion_hyperlinks': hyperlinks_count / total_tags, 
                    'html_number_js': scripts_count, 
                    }

        temp_df = pd.DataFrame(columns = ['url_length', 'url_num_segments', 'url_uri_count', 
                                    'text_length', 'text_num_words', 'text_noun_proportion', 'text_verb_proportion', 'text_adj_proportion', 
                                    'text_SMOG_index', 'text_FK_reading_level', 'html_num_unique_tags', 'html_proportion_hyperlinks', 'html_number_js'])
        
        
        temp_df = temp_df.append(temp_df_contents, ignore_index=True)
        temp_cv = CountVectorizer(analyzer='word', lowercase=False, vocabulary=vocabulary)

        temp_to_add = temp_cv.fit_transform([model_body])

        full_input_row = np.hstack((temp_df.values.astype(float), 
                                temp_to_add.toarray()
                        ))

        predictions.append(model.predict(full_input_row)[0])

    return predictions

if __name__ == "__main__": 
    os.chdir(sys.path[0])
    error_msg =  """Usage: \n\n
    python classify.py (dataset) (model) (input) (output) ... 
    \n\n
    (dataset) should be 'shalla' or 'dmoz'\n
    (model) should be 'url' or 'full'\n
    (output) should be either 'y' indicating save the output to a file in the same directory, or 'n' indicating
    to simply print the results to stdout\n
    (input) should similarly be 'y' or 'n', with 'y' indicating the URLs to be classified will be provided in a .csv file in the same directory, whose name
    will be the last argument in the command line. If 'no', then the remaining inputs on the command line should be the URLs to be classified.
    """

    which_list = '' 
    which_model = '' 
    which_output = ''
    which_input = '' 

    if len(sys.argv) < 6: 
        print(error_msg)
        sys.exit()
    if sys.argv[1].lower() not in ['dmoz', 'shalla']: 
        print(error_msg)
        sys.exit()
    if sys.argv[2].lower() not in ['url', 'full']: 
        print(error_msg)
        sys.exit()

    if sys.argv[3].lower() == 'y': 
        which_output = 'file' 
    elif sys.argv[3].lower() == 'n':
        which_output = 'stdout'
    else: 
        print(error_msg)
        sys.exit()

    if sys.argv[4].lower() == 'y': 
        which_input = 'file' 
    elif sys.argv[4].lower() == 'n':
        which_input = 'commandline'
    else: 
        print(error_msg)
        sys.exit()
    
    which_list = sys.argv[1]
    which_model = sys.argv[2]

    if which_model == 'url': 
        model_path = 'models/url_classifier_' + which_list +'.pickle'
        with open(model_path, 'rb') as f: 
            model = pickle.load(f)
        url_list = read_input(which_input)

        predictions = [] 

        for item in url_list: 
            # print("{}: {}".format(item, model.predict([process_url(item)])))
            if which_list == 'dmoz': # want the https protocols, etc.
                predictions.append(model.predict([process_url(item)])[0])
            elif which_list == 'shalla': 
                predictions.append(model.predict([remove_scheme(item)])[0])

        write_output(which_output, url_list, predictions)

    elif which_model == 'full': 
        with open('models/'+which_list+'_data/'+which_list+'_vocabulary.pickle', 'rb') as f: 
            vocabulary = pickle.load(f) 

        url_list = read_input(which_input) 

        with open('models/full_classifier_'+which_list+'.pickle', 'rb') as f: 
            curr_model = pickle.load(f)
        predictions = generate_prediction(url_list, curr_model)

        write_output(which_output, url_list, predictions)