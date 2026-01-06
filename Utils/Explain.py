from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings

def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.
    创建可解释性矩阵

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce

    例1: cat_emb_dim=1 & input_dim=3 & cat_idxs=[0,1,2] & post_embed_dim=3 -> all_emb_impact=[0,0,0] 
    -> indices_trick=[[0],[1],[2]] -> reducing_matrix=[[1,0,0],[0,1,0],[0,0,1]]

    
    例2: cat_emb_dim=[1,1,2] & input_dim=3 & cat_idxs=[0,1,2] & post_embed_dim=4 -> all_emb_impact=[0,0,1] 
    -> indices_trick=[[0],[1],[2,3]] -> reducing_matrix=[[1,0,0],[0,1,0],[0,0,1],[0,0,1]]
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1
    # 生成全0矩阵，shape：(post_embed_dim, input_dim)
    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1
    # 返回csc压缩的稀疏矩阵
    return scipy.sparse.csc_matrix(reducing_matrix)