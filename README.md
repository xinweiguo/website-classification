# website-classification
All of the relevant code produced for my senior CPSC 490 project at Yale, Spring 2021 towards Developing a Pipeline for Website Classification. 

Please feel free to reach out to me at danny.guo@yale.edu for any questions, comments, suggestions, or bug reports / difficulties. 

# Setup: 
First, run setup_files.py. This will, using filesplit https://pypi.org/project/filesplit/, stitch back together all of the models and data that were too large to be uploaded all at once into GitHub. 

At this point, classifier.py should immediately be runnable. 

# Description of each file

### generate_ground_truths.py

#### Usage: 
no additional command line arguments 

#### Description: Combines the broken up DMOZ (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OMV93V) and Shalla List (https://www.shallalist.de/) URL-category data sets together, 
and fetches the latest URL popularity data from Tranco (https://tranco-list.eu/), updated daily. Merges each of DMOZ and Shalla List with the popularity data set to generate a set of 
roughly 50,000 ground truths for URL categories, where the categories are determined by which data set used. These are then written to csv, and used in write_to_database.py

### write_to_database.py

#### Usage: 
python write_to_database.py (dataset) 

where

#### Description: 
Creates a database if one doesn't already exist, and tables for each of DMOZ and Shalla. 
