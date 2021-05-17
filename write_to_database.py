import os, sys
import sqlite3
import pandas as pd 
import numpy as np
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import trange 
import re
import nltk 
from nltk.tokenize import word_tokenize
from tqdm import trange
from word_segmentation import ngrams 
from string import digits

def process_url(url): 
    final_url = url 
    
    o = urlparse(url)
    if not o.scheme: 
        final_url = 'http://' + url 
        
    return final_url

def fetch_url_body(url): 
    full_url = process_url(url)
    
    res = requests.get(full_url, timeout = 8)

    output = ''

    # Initialize the object with the document
    soup = BeautifulSoup(res.text, "html.parser")

    # Get the whole body tag
    tag = soup.body

    # Print each string recursivey
#         if tag.strings: 
    for string in tag.strings:
        output += string

    return output

def fetch_url_raw(url): 
    full_url = process_url(url)

    res = requests.get(full_url, timeout = 10)

    return res.text

def parse_body(raw_body_text): 
    split_body = re.split('\W', raw_body_text)
    split_body = [x for x in split_body if x] 
    
    output = " ".join(split_body)
    
    output = output.lower()
    

    output = word_tokenize(output)

#     for word in output:
#         if word in stopwords.words('english'):
#             output.remove(word)
        
    return " ".join(output) 

def fetch_everything(url): 
    full_url = process_url(url)

    res = requests.get(full_url, timeout = 5)

    body_str = ''
    soup = BeautifulSoup(res.text, "html.parser")

    tag = soup.body

    for string in tag.strings:
        body_str += string

    return (parse_body(body_str), res.text)

if __name__ == "__main__": 
    os.chdir(sys.path[0])

    if not os.path.exists(os.getcwd()+'/databases/'):
        os.makedirs(os.getcwd()+'/databases/')

    which_list = ''

    if len(sys.argv) == 1: 
        print("Please enter as follows: python write_to_database.py <shalla|dmoz>")
        sys.exit()
    if sys.argv[1] == 'shalla': 
        which_list = 'shalla'
    elif sys.argv[1] == 'dmoz': 
        which_list = 'dmoz'
    else: 
        print("Please enter as follows: python write_to_database.py <shalla|dmoz>")
        sys.exit()

    df_path = 'data/' + which_list + '_popular.csv'
    df = pd.read_csv(df_path, header = 0) # first row (0) is Rank, Url, Category

    urls = df.Url
    categories = df.Category

    conn = sqlite3.connect('databases/url_data.db')
    cursor = conn.cursor()

    try: 
        query = """CREATE TABLE """ + which_list + """ (
            url TEXT, 
            category TEXT, 
            url_body LONGTEXT, 
            all_tags LONGTEXT
        );"""
        cursor.execute(query)

        conn.commit()
    except: 
        query = """DROP TABLE """ + which_list
        cursor.execute(query)

        query = """CREATE TABLE """ + which_list + """ (
            url TEXT, 
            category TEXT, 
            url_body LONGTEXT, 
            all_tags LONGTEXT
        );"""
        cursor.execute(query)
        conn.commit()

    try: 
        query = """CREATE TABLE """ + which_list + """_problem_urls (
            url TEXT, 
            reason TEXT
        );"""
        cursor.execute(query)

        conn.commit()
    except: 
        query = """DROP TABLE """ + which_list + """_problem_urls"""
        cursor.execute(query)

        query = """CREATE TABLE """ + which_list + """_problem_urls (
            url TEXT, 
            reason TEXT
        );"""

        cursor.execute(query)
        conn.commit()

    temp_url = ''
    temp_category = ''
    temp_url_body = ''
    temp__raw_html = '' 
    soup = '' 
    temp_all_tags = '' 

    for i in trange(len(urls)): 
        temp_category = categories[i]
        temp_url = urls[i]
        print(temp_url)
        
        try: 
            temp_result = fetch_everything(temp_url)
            temp_url_body = [word.lstrip(digits).rstrip(digits) for word in temp_result[0].split()]

            output = [] 
            try: 
                for item in temp_url_body: 
                    # print(ngrams.segment2(item))
                    for item2 in ngrams.segment2(item)[1]: 
                        if item2: 
                            output.append(item2) 
                temp_url_body = " ".join(output)
            except: 
                temp_url_body = " ".join(temp_url_body)
            
            temp_raw_html = temp_result[1]
            soup = BeautifulSoup(temp_raw_html, "html.parser")

            temp_all_tags = " ".join([tag.name for tag in soup.find_all()])
            
        except: 
            temp_url_body = ''
            temp_all_tags = ''

        try: # trying to write 
            if temp_url_body: 
                query = "INSERT INTO " + which_list+ " (url, category, url_body, all_tags) VALUES ('%s', '%s', '%s', '%s')" % (temp_url, temp_category, temp_url_body, temp_all_tags)
                cursor.execute(query)
            else: 
                query = "INSERT INTO " + which_list + "_problem_urls (url, reason) VALUES ('%s', '%s')" % (temp_url, 'fetch')
                cursor.execute(query)
        except: 
            query = "INSERT INTO " + which_list + "_problem_urls (url, reason) VALUES ('%s', '%s')" % (temp_url, 'write')
            cursor.execute(query)

        conn.commit()