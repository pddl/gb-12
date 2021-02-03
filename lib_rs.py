import numpy as np
from numpy import bincount, log, log1p, sqrt
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

################################################################################

def precision_at_k_(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    #print(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    
    return precision

def precision_at_k_score(df, actual=None, feature=None):
    r = df.apply(lambda x: precision_at_k_(x[feature], x[actual],  5), axis=1)
    return r.mean()

################################################################################

def data_to_sparse_matrix(data_train):
    user_item_matrix = pd.pivot_table(data_train, 
                                      index='user_id', columns='item_id', 
                                      values='quantity',
                                      aggfunc='sum', 
                                      fill_value=0,)

    user_item_matrix[user_item_matrix > 0] = 1        # так как в итоге хотим предсказать 
    user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

    userids = user_item_matrix.index.values
    itemids = user_item_matrix.columns.values

    matrix_userids = np.arange(len(userids))
    matrix_itemids = np.arange(len(itemids))

    id_to_itemid = dict(zip(matrix_itemids, itemids))
    id_to_userid = dict(zip(matrix_userids, userids))

    itemid_to_id = dict(zip(itemids, matrix_itemids))
    userid_to_id = dict(zip(userids, matrix_userids))

    # переведем в формат saprse matrix
    sparse_user_item = csr_matrix(user_item_matrix)
    
    return sparse_user_item, id_to_itemid, userid_to_id

################################################################################

def cosine_similarity(a, b):
    return 1 - cosine(a, b) # == (a*b).sum()/np.sqrt((a**2).sum())/np.sqrt((b**2).sum())

def pearson_similarity(x, y):
    return pearsonr(x, y)[0]

################################################################################

# site-packages/implicit/nearest_neighbours.py

def tfidf_weight_(X):
    """ Weights a Sparse Matrix by TF-IDF Weighted """
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X

def bm25_weight_(X, K1=100, B=0.8):
    """ Weighs each row of a sparse matrix X  by BM25 weighting """
    # calculate idf per term (user)
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # calculate length_norm per document (artist)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X

def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X

def cosine_weight_(X):
    return normalize(X)

################################################################################

def get_recommendations(model, sparse_user_item, userid=None, N=5, id2i=None, u2id=None):
    res = [id2i[rec[0]] for rec in 
                    model.recommend(userid=u2id[userid], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=True)]
    return res
