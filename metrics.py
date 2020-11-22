# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:03:28 2020

@author: Amogh
"""
def get_metrics(reconstructedTrain, testData):
    """
    calculates Root Mean Square Error(RMSE), Spearman Rank Correlation and Precision on top K. 

    Parameters
    ----------
    reconstructedTrain : 
        train data set.
    testData : 
        test data set.

    Returns
    -------
    RMSE : 
        Root Mean Square Error(RMSE).
    SRC : 
        Spearman Rank Correlation.
    PRECISION_TOP_K : 
        Precision on top K.

    """
    nUsers = len(reconstructedTrain)
    nItems = len(reconstructedTrain[0])
    squaredError, num_test = 0, 0
    for user in range(nUsers):
        for item in range(nItems):
            if testData[user, item] == 0:
                continue
            else:
                squaredError += (testData[user, item] - reconstructedTrain[user, item])**2
                num_test += 1
    
    SRC = 1 - ((6 * squaredError) / (num_test * (num_test**2 - 1)))
    RMSE = (squaredError / num_test) ** 0.5

    PRECISION_TOP_K = 0
    K = 10
    THRESHOLD = 3.5
    num_movies_rated = {}
    i = 0
    for user in testData:
        num_movies_rated[i] = user[user > 0].size
        i = i + 1
    i = 0
    all_precisions = []
    for user in reconstructedTrain:
        if num_movies_rated[i] < K:
            i = i + 1
            continue
        top_k_indices = (-user).argsort()[:K]
        top_k_values = [(index, user[index]) for index in top_k_indices]
        recommended = []
        for (index, user[index]) in top_k_values:
            if(user[index] >= 3.5):
                recommended.append((index, user[index]))
                if len(recommended) == K:
                    break
        count = 0
        for tup in recommended:
            if testData[i][tup[0]] >= THRESHOLD:
                count = count + 1
        if len(recommended) > 0:
            precision = count/len(recommended)
            all_precisions.append(precision)
        i = i + 1
    PRECISION_TOP_K = sum(all_precisions) / len(all_precisions)
    return RMSE, SRC, PRECISION_TOP_K
