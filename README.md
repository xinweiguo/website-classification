# website-classification
All of the relevant code produced for my senior CPSC 490 project at Yale, Spring 2021 towards Developing a Pipeline for Website Classification. 

Please feel free to reach out to me at danny.guo@yale.edu for any questions, comments, suggestions, or bug reports / difficulties. 

# Setup: 
First, run setup_files.py. This will, using filesplit https://pypi.org/project/filesplit/, stitch back together all of the models and data that were too large to be uploaded all at once into GitHub. 

At this point, classifier.py should immediately be runnable. More detailed descriptions of files can be found below. 

# Description of each file

### setup_files.py 

#### Usage: 
python setup_files.py (action)

where (action) should be either 'merge' or 'split'. Default is merge. 

#### Description: 
Combines together all of the split-up data for models, model inputs, and databases as GitHub per-file size limits prevent these from directly being uploaded to the repository. 

### generate_ground_truths.py

#### Usage: 
no additional command line arguments 

#### Description: Combines the broken up DMOZ (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OMV93V) and Shalla List (https://www.shallalist.de/) URL-category data sets together, 
and fetches the latest URL popularity data from Tranco (https://tranco-list.eu/), updated daily. Merges each of DMOZ and Shalla List with the popularity data set to generate a set of roughly 50,000 ground truths for URL categories, where the categories are determined by which data set used. These are then written to csv, and used in write_to_database.py

### write_to_database.py

#### Usage: 
python write_to_database.py (dataset) 

where (dataset) should be either 'dmoz' or 'shalla'

#### Description: 
Creates a database if one doesn't already exist, and a table for the specified data set. Fetches site content for specified data set and writes to database. The database content is retrieved in build_model_full.py for constructing the model. 

### build_model_url.py 

#### Usage: 
python build_model_url.py (dataset) 

where (dataset) should be either 'dmoz' or 'shalla'

#### Description: 
Creates a URL-feature-only classifier for the specified data set through an sklearn pipeline using Multinomial Naive Bayes and parameter-tuned using a Randomized Search Cross-validation. Uses keras for GPU-boosting if applicable. 

### build_model_full.py 

#### Usage: 
python build_model_full.py (dataset) 

where (dataset) should be either 'dmoz' or 'shalla'

#### Description: 
Creates a classifier for the specified data set through an sklearn pipeline using CountVectorizer, Tf-idf, and Random Forest Classifier that reads in the stored URL web content in the database (previously written to by the write_to_database.py script) and generates features described in the file. Additionally saves this model and model input to the /models/ subdirectory for ease of access. 

### classify.py 

#### Usage: 
python classify.py (dataset) (model) (input) (output) ... 

(dataset) should be 'shalla' or 'dmoz'
(model) should be 'url' or 'full'
(output) should be either 'y' indicating save the output to a file in the same directory, or 'n' indicating to simply print the results to stdout
(input) should similarly be 'y' or 'n', with 'y' indicating the URLs to be classified will be provided in a .csv file in the same directory, whose name will be the last argument in the command line. If 'n', then the remaining inputs on the command line should be the URLs to be classified.

#### Description: 




