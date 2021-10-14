#################
#### Imports ####
#################

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.float_format', lambda x: '%.5f' % x)




#################
#### Acquire ####
#################

def acquire_data():
    """
    Function reads csv from directory and returns as a dataframe
    """
    return pd.read_csv('world_data.csv')

def df_info(df):
    """
    Function returns info about a dataframe and its features. It will return a lot of information.
    """
    print(df.info())
    print()
    print(df.size, df.shape)
    print()
    print(df.isnull().sum())
    print()
    for col in df.columns:
        print(col)
        print('Unique values:', df[col].nunique())
        print()
        print('Datatype:', df[col].dtype)
        print()
        print(df[col].value_counts())
        print('|--------------------|')
        print('|--------------------|')
        print('|--------------------|')
    return df.describe().T

def plot_dist(df):
    for col in df.columns:
        sns.distplot(df[col])
        plt.title(col)
        plt.show()


#################
#### Prepare ####
#################

def prep_data(df):
    """
    Function takes in dataframe:
        1. Creates 3 new cols
        2. Converts new minutes col to categorical
        3. Creates dummy cols
        4. Drops minutes and other unused cols
    """
    # create new cols
    df['name_length'] = [len(name) for name in df.artist_name]
    df['song_name_length'] = [len(track) for track in df.track_name]
    df['minutes'] = (df.duration_ms / 60000)
    # bin to create categorical 'minutes' column
    df['minutes'] = pd.cut(df.minutes, bins=[0, 1, 2, 3, 4, 5, 25])
    
    #Create dummy df for regression
    dummy_df = pd.get_dummies(df['minutes'], dummy_na=False, drop_first=False)
    # Concatenate the dummy_df dataframe above with the original df
    df = pd.concat([df, dummy_df], axis=1)
    
    # Drop unused columns
    df.drop(columns=['Unnamed: 0', 'artist_name', 'track_name', 'track_id'], inplace=True)
    
    return df

def train_validate_test_split(df, seed=117):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test