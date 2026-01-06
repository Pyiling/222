from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

def categorical_embeddings(train, target, limit=200):
    """
    得到哪些离散特征需要做embedding
    train: pandas数组
    target: 标签列名
    limit: 判定是否为离散变量
    """
    # 得到每个特征的取值分布和类型
    nunique = train.nunique()
    types = train.dtypes

    # 分别记录每个类别特征的名称和类别个数
    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < limit:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            # train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        # else:
        #     train.fillna(train.loc[train_indices, col].mean(), inplace=True)
    
    # 得到所有特征的名称list
    features = [ col for col in train.columns if col not in [target]] 

    # 得到类别特征的index
    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    # 得到类别特征的对应类别个数
    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return cat_idxs, cat_dims

def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    """
    Check parameters related to embeddings and rearrange them in a unique manner.
    检查embedding的参数,并根据输入来规范化
    """
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims