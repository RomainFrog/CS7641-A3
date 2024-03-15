"""
This file contains all auxiliary functions for loading datasets
and preprocessing them before importing them into the notebooks.

Each Function should have the name load_<dataset_name> and return
a pandas dataframe.

Each dataset is a classification problem. The target variable
is always called 'target' and is the last column of the dataframe.

Romain Froger, 2024
Georgia Institute of Technology
All rights reserved
"""

import pandas as pd


def load_wine():
    """
    Load wine dataset from sklearn
    """
    from sklearn.datasets import load_wine

    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df

def load_titanic():
    """
    Load titanic dataset from seaborn
    """
    import seaborn as sns

    df = sns.load_dataset('titanic')
    # remove the deck column
    df = df.drop('deck', axis=1)
    # remove people who no age
    df = df.dropna(subset=['age'])
    # drop all remaining missing values
    df = df.dropna()
    # rename alive column to target and put it at the end
    df = df.rename(columns={'alive': 'target'})
    # convert target to 0 and 1
    df['target'] = df['target'].map({'yes': 1, 'no': 0})
    # convert alone to 0 and 1
    df['alone'] = df['alone'].map({True: 1, False: 0})
    # drop adult_male column
    df = df.drop('adult_male', axis=1)
    # drop class column
    df = df.drop('class', axis=1)
    # drop who column
    df = df.drop('who', axis=1)
    # create adult column based on age (int)
    df['adult'] = df['age'].apply(lambda x: 1 if x >= 18 else 0)
    # drop embarked column
    df = df.drop('embarked', axis=1)

    # translate embarked_town to one hot encoding (int)
    embark_town = pd.get_dummies(df['embark_town'])
    df = df.join(embark_town)
    df = df.drop('embark_town', axis=1)
    # cast Cherbourg, Queenstown, Southampton to int
    df['Cherbourg'] = df['Cherbourg'].astype(int)
    df['Queenstown'] = df['Queenstown'].astype(int)
    df['Southampton'] = df['Southampton'].astype(int)

    # translate sex to one hot encoding (int)
    sex = pd.get_dummies(df['sex'])
    df = df.join(sex)
    df = df.drop('sex', axis=1)

    # cast sex to int
    df['male'] = df['male'].astype(int)
    df['female'] = df['female'].astype(int)

    # drop survived column
    df = df.drop('survived', axis=1)

    # reset index
    df = df.reset_index(drop=True)

    # 
    return df


def load_wine_quality():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    wine_quality = fetch_ucirepo(id=186) 
    
    # data (as pandas dataframes) 
    X = wine_quality.data.features 
    y = wine_quality.data.targets 

    # create new target column (0 if quality <= 5, 1 if quality > 5)
    target = y.map(lambda x: 0 if x <= 5 else 1)
    X['target'] = target
  

    return X


def load_abalone():
    """
    Load abalone dataset from UCI
    """
    from ucimlrepo import fetch_ucirepo

    abalone = fetch_ucirepo(id=1)
    df = abalone.data.features
    df['target'] = abalone.data.targets
    return df

def load_mushroom():
    """
    Load mushroom dataset from UCI
    """
    from ucimlrepo import fetch_ucirepo

    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.features
    df['target'] = mushroom.data.targets
    return df

def load_diabetes():
    """
    Load diabetes dataset from UCI
    """
    path = "data/diabetes.csv"
    df = pd.read_csv(path)
    #rename outcome column in target
    df = df.rename(columns={'Outcome': 'target'})
    return df


def load_beans():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    dry_bean_dataset = fetch_ucirepo(id=602) 
    
    # data (as pandas dataframes) 
    X = dry_bean_dataset.data.features 
    y = dry_bean_dataset.data.targets 
    
    X['target'] = y

    return X

def load_hand_written_digits():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    digits = fetch_ucirepo(id=80) 
    
    # data (as pandas dataframes) 
    X = digits.data.features 
    y = digits.data.targets 
    
    X['target'] = y

    return X
