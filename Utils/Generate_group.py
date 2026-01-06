from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings

def check_list_groups(list_groups, input_dim):
    """
    检查输入的group是否规范
    list_groups: 输入的group,形式一般为list套list:[[1,2,3], [4,5,6]]
    input_dim: 输入的特征维度
    """
    # 首先检查输入的group是否为list
    assert isinstance(list_groups, list), "list_group must be a list of list."

    #如果list为空，则返回
    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."
            # 判断是否符合list套list
            assert isinstance(group, list), msg
            # 判断单个group list是否为空
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    # 统计list_groups里的元素个数
    n_elements_in_groups = np.sum([len(group) for group in list_groups])

    # 统计独一无二的元素个数,并判断是否有重复元素
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = f"One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    # 判断是否有元素越界
    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."

    return

def create_group_matrix(list_groups, input_dim):
    """
    创建一个group矩阵
    list_groups: 输入的原始group
    input_dim: 输入的特征维度
    eg. input_dim = 6, list_groups = [[1,2,3],[4,5,6]] => group_matrix = [[1/3,1/3,1/3,0,0,0],[0,0,0,1/3,1/3,1/3]]
    """
    # 首先检查手否规范
    check_list_groups(list_groups, input_dim)

    # 如果输入的list为空，则代表每个特征都各自为一个group，创建一个eye矩阵
    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        # 计算得到group的个数，由于存在不在list_group的元素，意味着那些元素也各自成一个group，计算时需要考虑到
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        # 新建一个维度大小为（n_groups, input_dim）的空矩阵
        group_matrix = torch.zeros((n_groups, input_dim))

        # 得到每个特征的index
        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # 添加每个group里元素的权重，初始权重平分
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # 同时移除已添加元素的index
                remaining_features.remove(elem_idx)
            # 迭代到下一个group
            current_group_idx += 1
        # 未在list_groups里面的元素自成一个group
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix