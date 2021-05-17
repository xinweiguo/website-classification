import urllib.request
import zipfile 
from datetime import date 
import pandas as pd 
pd.options.mode.chained_assignment = None # default = 'warn'
import os, sys

if __name__ == "__main__": 
    os.chdir(sys.path[0])
    url = 'https://tranco-list.eu/top-1m.csv.zip'

    # if not os.path.exists(os.getcwd()+'/data/'):
    #     os.makedirs(os.getcwd()+'/data/')

    remote = urllib.request.urlopen(url)  # read remote file
    data = remote.read()  # read from remote file
    remote.close()  # close urllib request

    tranco_dir = os.getcwd() + '/data/' + str(date.today().month) + '_' + str(date.today().day) + '_' + str(date.today().year)+'/'
    if not os.path.exists(tranco_dir):
        os.makedirs(tranco_dir)

    local = open(tranco_dir+'tranco.zip', 'wb')  # write binary to local file
    local.write(data)
    local.close()  # close file

    with zipfile.ZipFile(tranco_dir+'tranco.zip',"r") as zip_ref:
        zip_ref.extractall(tranco_dir)

    dmoz_df_1 = pd.read_csv(os.getcwd() + '/data/dmoz_1.csv', header=None)
    dmoz_df_1.columns = ['Url', 'Category']
    dmoz_df_2 = pd.read_csv(os.getcwd() + '/data/dmoz_2.csv', header=None)
    dmoz_df_2.columns = ['Url', 'Category']
    dmoz_df_3 = pd.read_csv(os.getcwd() + '/data/dmoz_3.csv', header=None)
    dmoz_df_3.columns = ['Url', 'Category']
    dmoz_df_4 = pd.read_csv(os.getcwd() + '/data/dmoz_4.csv', header=None)
    dmoz_df_4.columns = ['Url', 'Category']

    dmoz_df = pd.concat([dmoz_df_1, dmoz_df_2, dmoz_df_3, dmoz_df_4], ignore_index=True)

    dmoz_short = dmoz_df.loc[dmoz_df.loc[:, 'Url'].str.startswith("http://www.").fillna(False), :]
    dmoz_short.loc[:, 'Url'] = dmoz_short.loc[:, 'Url'].str.replace("http://www.", "").str[:-1]

    dmoz_short = dmoz_short.drop_duplicates(subset=['Url'], keep='first')
    dmoz_short.reset_index(inplace=True, drop=True)

    popularity_df = pd.read_csv(tranco_dir+'top-1m.csv', header = None)
    popularity_df.columns = ['Rank', 'Url']

    dmoz_popular = popularity_df.merge(dmoz_short, on = 'Url', how = 'inner')


    shalla_df_1 = pd.read_csv(os.getcwd() + '/data/shalla_1.csv', header=None)
    shalla_df_1.columns = ['Url', 'Category']
    shalla_df_2 = pd.read_csv(os.getcwd() + '/data/shalla_2.csv', header=None)
    shalla_df_2.columns = ['Url', 'Category']
    shalla_df_3 = pd.read_csv(os.getcwd() + '/data/shalla_3.csv', header=None)
    shalla_df_3.columns = ['Url', 'Category']
    shalla_df_4 = pd.read_csv(os.getcwd() + '/data/shalla_4.csv', header=None)
    shalla_df_4.columns = ['Url', 'Category']

    shalla_df = pd.concat([shalla_df_1, shalla_df_2, shalla_df_3, shalla_df_4], ignore_index=True)
    shalla_df.columns = ['Url', 'Category']
    shalla_popular = popularity_df.merge(shalla_df, on = 'Url', how = 'inner')

    dmoz_df.to_csv(os.getcwd()+'/data/dmoz.csv', header=False, index=False)
    shalla_df.to_csv(os.getcwd()+'/data/shalla.csv', header=False, index=False)

    dmoz_popular.to_csv(os.getcwd()+'/data/dmoz_popular.csv', index=False)
    shalla_popular.to_csv(os.getcwd()+'/data/shalla_popular.csv', index=False)
