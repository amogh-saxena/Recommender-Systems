# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:03:28 2020

@author: Amogh
"""
"""
store the train and test dataset as numpy arrays.
"""
import pandas as pd
import numpy as np


def change(data, min_user_index, max_user_index, nb_movies):
    """
    converts the raw data to numpy array

    Returns
    -------
    new_data : TYPE
        train data as numpy array.

    """
    new_data = []
    for id_users in range(min_user_index, max_user_index + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

      
def main():
    print("[INFO] reading CSV files")
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    training_set = np.array(training_set, dtype='int')
    test_set = np.array(test_set, dtype='int')
    nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    min_user_index = 1
    max_user_index = nb_users
    training_set = change(training_set, min_user_index, max_user_index, nb_movies)
    test_set = change(test_set, min_user_index, max_user_index, nb_movies)
    training_data = np.array([np.array(x) for x in training_set])
    test_data = np.array([np.array(x) for x in test_set])

    print("[INFO] Saving as NPY files")
    np.save('train.npy',training_data)
    np.save('test.npy', test_data)
    
if __name__ == '__main__':
    main()
    
    
    
    
    