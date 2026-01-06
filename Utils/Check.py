from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings

def check_input(X):
    """
    Raise a clear error if X is a pandas dataframe
    and check array according to scikit rules
    检查输入是否为需要的类型，一般是numpy数组
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        err_message = "Pandas DataFrame are not supported: apply X.values when calling fit"
        raise TypeError(err_message)
    check_array(X, accept_sparse=True)


def check_warm_start(warm_start, from_unsupervised):
    """
    Gives a warning about ambiguous usage of the two parameters.
    检查是否热启动，若是热启动则先警告一下
    """
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg = "warm_start will be ignore, training will start from unsupervised weights"
        warnings.warn(warn_msg)
    return

def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    检查eval_set是否和训练集相匹配

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    # 获取每个集合的名称
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    # 检查set和name长度是否匹配
    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    # 检查set里每个turtle都有两个变量，即x和y
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    # 对set里每个集合进行检查
    for name, (X, y) in zip(eval_name, eval_set):
        # 检查x是否和X_train的维度一致
        check_input(X)
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg
        # 检查y是否和y_train的维度一致
        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg
      # 检查x是否和X_train的特征数是否一致
        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        # 检查x和y的样本数是否匹配
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu
    设定用cpu还是gpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)