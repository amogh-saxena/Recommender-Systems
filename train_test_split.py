# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:03:28 2020

@author: Amogh
"""


import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    """
    function to handle test/train splitting

    splits the raw dataset into train and test sets, and store them as csv files.
    """
    ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None,
                      engine = 'python', encoding = 'latin-1')
    training_data, test_data = train_test_split(ratings, test_size = 0.2, random_state = 0)
    training_data.to_csv('train.csv', header = None, index = False)
    test_data.to_csv('test.csv', header = None, index = False)
    
if __name__ == '__main__':
    main()