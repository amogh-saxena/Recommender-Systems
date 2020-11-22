# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:03:28 2020

@author: Amogh
"""
import numpy as np
from math import sqrt
from svd import build_svd_matrices
from svd import top_90_energy
from metrics import get_metrics
import timeit
from sklearn.metrics import mean_squared_error

def C_U_R(k):
    """
    Performs the CUR decomposition on the user matrix and stores them as numpy arrays

    Parameters
    ----------
    k : 
       numbers of rows and colums taken

    Returns
    -------
    None.

    """
    user_movie_matrix = np.load('train.npy')
    
    total_ssq = 0 #sum of squares of all elements
    n_users = user_movie_matrix.shape[0]
    n_users = user_movie_matrix[0].size
    for i in range(n_users):
        for j in range(n_users):
            total_ssq = total_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
    prob_users = []
    prob_movies = []
    for i in range(n_users):
        row_ssq = 0 #
        for j in range(n_users):
            row_ssq = row_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        prob_users.append(row_ssq/total_ssq)#computing user probabilities
    for j in range(n_users):
        col_ssq = 0
        for i in range(n_users):
            col_ssq = col_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        prob_movies.append(col_ssq/total_ssq)#computing movie probabilties
    top_users = np.random.choice(len(prob_users),k, replace=False, p=prob_users) #sampling rows
    top_movies = np.random.choice(len(prob_movies),k, replace=False, p=prob_movies) #sampling columns
    top_movies.sort()
    top_users.sort()
    C = []
    R = []
    for i in top_users:
        R.append(list(user_movie_matrix[i]/sqrt(k*prob_users[i])))
    for j in top_movies:
        C.append(list(user_movie_matrix[:,j]/sqrt(k*prob_movies[j])))
    Ct = np.transpose(C)
    W = []
    for i in top_users:
        X=[]
        for j in top_movies:
            X.append(user_movie_matrix[i][j])#intersection of sampled rows and columns
        W.append(np.array(X))
    W = np.array(W)
    x,yt,sigma = build_svd_matrices(W)#SVD of intersection
    pinv_sigma = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
    sig_sq = np.linalg.matrix_power(pinv_sigma, 2)#square of pseudo-inverse
    y = np.transpose(yt)
    xt = np.transpose(x)
    
    
    
    U = np.matmul(y, sig_sq)
    U = np.matmul(U, xt)    #reconstructing U
    
    
    np.save('cur_ct.npy', Ct)
    np.save('cur_r.npy', R)
    new_x, new_yt, new_sigma = top_90_energy(x,yt,sigma)
    
    
    pinv_new_sigma = np.linalg.pinv(new_sigma)
    new_sig_sq = np.linalg.matrix_power(pinv_new_sigma, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    
    
    U = np.matmul(y, new_sig_sq)
    U = np.matmul(U, xt)
    np.save('cur_u.npy', U)
    
def C_U_R_90(k):
    """
    Performs the CUR decomposition on the user matrix with 90% retained energy and stores them as numpy arrays

    Parameters
    ----------
    k : 
        numbers of rows and colums taken

    Returns
    -------
    None.

    """
    user_movie_matrix = np.load('train.npy')
    
    total_ssq = 0 #sum of squares of all elements
    n_users = user_movie_matrix.shape[0]
    n_users = user_movie_matrix[0].size
    for i in range(n_users):
        for j in range(n_users):
            total_ssq = total_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
    prob_users = []
    prob_movies = []
    for i in range(n_users):
        row_ssq = 0 #
        for j in range(n_users):
            row_ssq = row_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        prob_users.append(row_ssq/total_ssq)#computing user probabilities
    for j in range(n_users):
        col_ssq = 0
        for i in range(n_users):
            col_ssq = col_ssq + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        prob_movies.append(col_ssq/total_ssq)#computing movie probabilties
    top_users = np.random.choice(len(prob_users),k, replace=False, p=prob_users) #sampling rows
    top_movies = np.random.choice(len(prob_movies),k, replace=False, p=prob_movies) #sampling columns
    top_movies.sort()
    top_users.sort()
    C = []
    R = []
    for i in top_users:
        R.append(list(user_movie_matrix[i]/sqrt(k*prob_users[i])))
    for j in top_movies:
        C.append(list(user_movie_matrix[:,j]/sqrt(k*prob_movies[j])))
    Ct = np.transpose(C)
    W = []
    for i in top_users:
        X=[]
        for j in top_movies:
            X.append(user_movie_matrix[i][j])#intersection of sampled rows and columns
        W.append(np.array(X))
    W = np.array(W)
    x,yt,sigma = build_svd_matrices(W)#SVD of intersection
    pinv_sigma = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
    sig_sq = np.linalg.matrix_power(pinv_sigma, 2)#square of pseudo-inverse
    y = np.transpose(yt)
    xt = np.transpose(x)
    U = np.matmul(y, sig_sq)
    U = np.matmul(U, xt)    #reconstructing U
    np.save('cur_ct_90.npy', Ct)
    np.save('cur_r_90.npy', R)
    new_x, new_yt, new_sigma = top_90_energy(x,yt,sigma)#SVD with top 90% energy
    pinv_new_sigma = np.linalg.pinv(new_sigma)
    new_sig_sq = np.linalg.matrix_power(pinv_new_sigma, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    U = np.matmul(y, new_sig_sq)
    U = np.matmul(U, xt)
    np.save('cur_u_90.npy', U) 
    
def srcr(matrix,final):
    """
    Calculates Spearman Rank Correlation

    Parameters
    ----------
    matrix : 
        train values matrix
    final : 
        Multiplication of C,U,R matrices.

    Returns
    -------
    val : 
        Spearman Rank Correlation.

    """
    count=0
    sum=0
    for i in range(0,len(matrix)):
	    for j in range(0,len(matrix[i])):
		    sum=sum+(matrix[i][j]-final[i][j])**2
		    count=count+1
    sum=6*sum
    temp=(count**3)-count
    val=1-(sum/temp)
    return val
	
	
#calculating precision on top k for CUR
def ponk_cur(mat, final):
  """
  calculating precision on top k for CUR   

    Parameters
    ----------
    mat : 
        train values matrix.
    final : 
        Multiplication of C,U,R matrices.

    Returns
    -------
    TYPE
        precision on top k.

  """
  k_mat=final.tolist()
  count=0.00
  match=0.00
  for i in range(0,len(mat)):
	  for j in range(0,len(mat[i])):
		  count=count+1
		  a=int(round(mat[i][j]))
		  b=int(round(k_mat[i][j]))
		  if (a==b):
			  match=match+1
  precision=(match*100)/count
  return precision/100

def main():
    start=timeit.default_timer()
    C_U_R(600)
    print("Time taken")
    stop=timeit.default_timer()
    print("%s seconds" %(stop-start))
    C_U_R_90(600)
    print("Time taken for 90%")
    stop=timeit.default_timer()
    print("%s seconds" %(stop-start))
    Ct = np.load('cur_ct.npy')
    A = np.load('train.npy')
   
    R = np.load('cur_r.npy')
    U = np.load('cur_u.npy')
    final = np.matmul(Ct, U)
    final = np.matmul(final, R)
    rmse_err=sqrt(mean_squared_error(A, final))
    print("RMSE error is :")
    print(rmse_err)
    print("Precision on top k is :")
    ans=ponk_cur(A, final)
    print(ans)
    answer = srcr(A, final)
    print("Spearman Rank Correlation is ", answer)
    Ct_90 = np.load('cur_ct_90.npy')
    R_90 = np.load('cur_r_90.npy')
    U_90 = np.load('cur_u_90.npy')
    final_90 = np.matmul(Ct_90, U_90)
    final_90 = np.matmul(final_90, R_90)
    rmse_err_90=sqrt(mean_squared_error(A, final_90))
    print("RMSE error for 90% is :")
    print(rmse_err_90)
    print("Precision on top k for 90% is :")
    ans_90=ponk_cur(A, final_90)
    print(ans_90)
    answer_90 = srcr(A, final_90)
    print("Spearman Rank Correlation for 90%  is ", answer_90)
    del A
        
if __name__ == '__main__':
    main()