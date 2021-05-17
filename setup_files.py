from fsplit.filesplit import Filesplit
import os, sys

def split_cb(f, s):
    print("file: {0}, size: {1}".format(f, s))

def merge_cb(f, s):
    print("file: {0}, size: {1}".format(f, s))

if __name__ == "__main__": 
    os.chdir(sys.path[0])
    fs = Filesplit()

    # files to split / merge: 
    # url models - shalla and dmoz. Located at 
    # full models - shalla and dmoz. Located at 
    # full inputs - shalla and dmoz. Located at 

    action = '' 

    if len(sys.argv) == 1: # no arguments provided, default action will be to merge everything back together 
        action = 'merge'
    elif sys.argv[1] == 'split': # explicitly state split 
        action = 'split'
    elif sys.argv[1] == 'merge': 
        action = 'merge'
    else: 
        print("Please enter either no arguments, or either split or merge")
        sys.exit()

    
    size_limit = 25000000

    if action == 'split': 
        fs.split(file="models/url_classifier_dmoz.pickle", split_size=size_limit, output_dir="models/url_classifier_dmoz_split", callback=split_cb)
        fs.split(file="models/url_classifier_shalla.pickle", split_size=size_limit, output_dir="models/url_classifier_shalla_split", callback=split_cb)
        fs.split(file="models/full_classifier_dmoz.pickle", split_size=size_limit, output_dir="models/full_classifier_dmoz_split", callback=split_cb)
        fs.split(file="models/full_classifier_shalla.pickle", split_size=size_limit, output_dir="models/full_classifier_shalla_split", callback=split_cb)
        fs.split(file="models/dmoz_data/dmoz_full_x.npy", split_size=size_limit, output_dir="models/dmoz_data/dmoz_full_x_split", callback=split_cb)
        fs.split(file="models/shalla_data/shalla_full_x.npy", split_size=size_limit, output_dir="models/shalla_data/shalla_full_x_split", callback=split_cb)
        fs.split(file="databases/url_data.db", split_size=size_limit, output_dir="databases/url_data_split", callback=split_cb)

    elif action == 'merge': 
        fs.merge(input_dir="models/url_classifier_dmoz_split", output_file = 'models/url_classifier_dmoz.pickle',callback=merge_cb)
        fs.merge(input_dir="models/url_classifier_shalla_split", output_file = 'models/url_classifier_shalla.pickle',callback=merge_cb)
        fs.merge(input_dir="models/full_classifier_dmoz_split", output_file = 'models/full_classifier_dmoz.pickle',callback=merge_cb)
        fs.merge(input_dir="models/full_classifier_shalla_split", output_file = 'models/full_classifier_shalla.pickle',callback=merge_cb)
        fs.merge(input_dir="models/dmoz_data/dmoz_full_x_split", output_file = 'models/dmoz_data/dmoz_full_x.npy',callback=merge_cb)
        fs.merge(input_dir="models/shalla_data/shalla_full_x_split", output_file = 'models/shalla_data/shalla_full_x.npy',callback=merge_cb)
        fs.merge(input_dir="databases/url_data_split", output_file = 'databases/url_data.db',callback=merge_cb)