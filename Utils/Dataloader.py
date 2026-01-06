from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings

class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


class SparseTorchDataset(Dataset):
    """
    Format for csr_matrix
    # 将稀疏矩阵转为dataset

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        y = self.y[index]
        return x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x


class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix
    # 将稀疏矩阵转为dataset

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        return x

def create_sampler(weights, y_train):
    """

        这将根据给定的权重创建一个采样器。
        解决类别不平衡问题，为每个样本分配采样权重。

        参数：
        weights：0、1、字典或可迭代对象
        如果是0（默认值）：不应用权重
        如果是1：仅用于分类，将平衡类别的频率
        如果是字典：键是对应的类别值，值是样本权重
        如果是可迭代对象：列表或np数组的长度必须等于训练集中的元素数量
        y_train：np.array
        训练目标
    """
    if isinstance(weights, int):
        # 如果weight=0，则意味着不进行随机采样，利用整个原有数据集进行训练，并且打乱顺序
        if weights == 0:
            need_shuffle = True
            sampler = None
        # 如果weight=1，则意味着利用每个类的样本数进行权重分配，同类样本的权重一致，均为样本量的倒数，并且不打乱，意义在于数量越少的类别权重会越大，采样被选中的概率也越大（可重复），使得最终采样得到的数据集中每个类的样本数尽可能一致
        elif weights == 1:
            need_shuffle = False
            # 统计每个类的样本量
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )
            # 利用倒数计算权重
            weights = 1.0 / class_sample_count
            # 利用循环对每个样本分配权重
            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    # 如果weights为dict，则为自定义每个类的权重
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    # 如果weights为list，且长度为训练集样本总量，则意味着为每个样本客制化权重
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler

def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    创建数据加载器，根据权重和是否平衡进行子采样。

    参数：
    X_train：np.ndarray
    训练数据
    y_train：np.array
    映射后的训练目标
    eval_set：元组列表
    评估元组集合列表（X，y）
    weights：0、1、字典或可迭代对象
    如果是0（默认值）：不应用权重
    如果是1：仅用于分类，将使用倒数频率平衡类别
    如果是字典：键是对应的类别值，值是样本权重
    如果是可迭代对象：列表或np数组的长度必须等于训练集中的元素数量
    batch_size：int
    每个批次加载多少样本
    num_workers：int
    用于数据加载的子进程数量。0表示数据将在主进程中加载
    drop_last：bool
    设置为True以丢弃最后一个不完整的批次，如果数据集大小不是批次大小的整数倍。如果为False，且数据集大小不是批次大小的整数倍，则最后一个批次将更小
    pin_memory：bool
    在训练期间是否将GPU内存固定

    返回：
    train_dataloader, valid_dataloader：torch.DataLoader, torch.DataLoader
    训练和验证数据加载器
    """
    # 根据weights来选择sample类型以及是否打乱
    need_shuffle, sampler = create_sampler(weights, y_train)
    # 针对不同类型的矩阵数据创建训练集dataloader
    if scipy.sparse.issparse(X_train):
        train_dataloader = DataLoader(
            SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = DataLoader(
            TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    # 验证集同上，但验证集不涉及采样和打乱，注意：采样只在训练集的dataloader中进行，目的在于增强模型的泛化性能
    # 验证集可能有多个，可能包含训练集为了获得训练集的准确率
    valid_dataloaders = []
    for X, y in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        else:
            valid_dataloaders.append(
                DataLoader(
                    TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    return train_dataloader, valid_dataloaders