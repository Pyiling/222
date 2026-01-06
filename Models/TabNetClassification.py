import os
import sys
sys.path.append('/workspace')
from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy.sparse import csc_matrix
import Models.TabNet as TabNet
from Utils.Embedding import check_embedding_parameters
from Utils.Check import check_input, check_warm_start, validate_eval_set, define_device, ComplexEncoder
from Utils.MultiClass import infer_output_dim, check_output_dim
from Utils.Dataloader import create_dataloaders, SparsePredictDataset, PredictDataset
from Utils.Generate_group import create_group_matrix
from Utils.Explain import create_explain_matrix
from Utils.Metrics import check_metrics, MetricContainer
from Utils.Callbacks import History, EarlyStopping, LRSchedulerCallback, CallbackContainer
# from sklearn.base import BaseEstimator

from torch.utils.data import DataLoader
import io
import json
from pathlib import Path
import shutil
import zipfile
import warnings
import copy
import scipy
from scipy.special import softmax
from sklearn.base import BaseEstimator
'''
BaseEstimator 是 scikit-learn 中的一个基类，它通常用于自定义估计器（estimator）。
估计器是 scikit-learn 中的核心概念，它表示一个能够训练和应用模型的对象。
通过继承 BaseEstimator，你的自定义估计器将获得一些通用方法和属性，
如 fit（用于训练模型）、predict（用于进行预测）、get_params（用于获取估计器的参数）等。
'''

@dataclass
class TabNetClassifier(BaseEstimator): # 自定义估计器通常需要继承自 BaseEstimator
    """
    TabNet分类器 解决分类问题
    """
    # 初始化决策层维度
    n_d: int = 8
    # 初始化注意层维度
    n_a: int = 8
    # 初始化step数，即循环多少次
    n_steps: int = 3

    # Prior 在 TabNet 中是指预训练的特征选择权重，用于帮助模型在训练期间进行特征选择。
    # gamma 参数的值范围通常设置为1到2之间，用于控制 prior 权重的更新。
    gamma: float = 1.3

    '''
        离散类别特征（Categorical Features）是数据集中包含的一种特征类型，
        其值来自于一个有限的离散集合，通常代表了某种分类或类别。
        这些特征描述了某些事物的类别或标签，而不是连续的数值。
    '''
    # 初始化离散类别特征的index列表
    cat_idxs: List[int] = field(default_factory=list)
    # 初始化离散类别特征的取值个数list
    cat_dims: List[int] = field(default_factory=list)
    ''' 
        初始化离散分类变量嵌入后的维度
        嵌入维度是一个在深度学习中常见的概念，它用于将离散类别特征映射到连续的向量空间中，以便神经网络可以处理它们。
        这个值表示每个离散类别特征将被映射到一个具有多少维度的连续向量空间。
    '''
    cat_emb_dim: int = 1

    # GLU 是一种神经网络中的激活函数
    # 初始化独立的GLU_Block的数量
    # 独立的 GLU Block 通常用于处理不同特征的信息，可以捕获不同特征之间的关系。
    n_independent: int = 2
    # 初始化共享的GLU_Block的数量
    # 共享的 GLU Block 用于捕获特征之间的交叉关系，有助于处理特征之间的相互作用。
    n_shared: int = 2
    # 防止出现log(0)的情况，不可导
    epsilon: float = 1e-15
    # BN层中滑动平均系数
    # BN层用于对网络中间层的输入进行批量标准化。是一种常用的正则化方法，旨在加速神经网络的训练过程并提高模型的性能。
    momentum: float = 0.02
    # 稀疏性损失系数
    # 使得模型在训练期间选择少数重要的特征，从而提高模型的可解释性和泛化能力。
    # 稀疏性损失的作用是通过正则化约束来控制模型选择哪些特征用于预测任务。
    lambda_sparse: float = 1e-3
    # 初始化随机种子
    seed: int = 0 # 确保在不同运行中获得相同的结果
    # 是否进行梯度裁剪，防止梯度 爆炸
    # 梯度爆炸指的是在训练神经网络时，梯度值变得非常大，导致数值不稳定性和训练失败。
    clip_value: int = 1
    # 输出迭代过程的信息
    # 该参数的值通常是一个整数，它决定了模型训练过程中生成的日志信息的详细程度。
    verbose: int = 1
    # 初始化优化器为Adam 需要设置一些超参数，如学习率（learning rate）和权重衰减（weight decay）
    # 优化器（Optimizer）是用于更新模型参数以最小化损失函数的工具
    optimizer_fn: Any = torch.optim.Adam
    # 初始化优化器参数dict，学习率等
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2))
    # 对学习率进行更新，默认不更新
    scheduler_fn: Any = None
    # 学习率更新参数dict
    scheduler_params: Dict = field(default_factory=dict)
    # mask保持稀疏性的函数类型
    mask_type: str = "sparsemax"
    # 输入维度
    input_dim: int = None
    # 输出维度
    output_dim: int = None
    # 运行设备名
    device_name: str = "auto"
    n_shared_decoder: int = 1
    n_indep_decoder: int = 1
    # 初始化group列表
    grouped_features: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        # These are default values needed for saving model
        # 这些是用于保存模型所需的默认值。
        # 初始化bs大小
        self.batch_size = 1024
        '''
        批量大小是在训练深度学习模型时，每个批次（batch）中包含的样本数量。
        在这里，批量大小被设置为 1024，这意味着每个训练迭代中将使用 1024 个样本来更新模型的参数。
        '''
        # 初始化GNB中的虚拟BS大小
        self.virtual_batch_size = 128
        '''
        虚拟批量大小通常与批量大小一起使用，用于一种称为批规范化（Batch Normalization）的技术。
        虚拟批量大小指示在批规范化中使用的批次大小。
        在这里，虚拟批量大小被设置为 128，表示在批规范化中将使用虚拟批量大小为 128 的批次。
        批规范化是一种用于加速深度学习模型训练和提高模型性能的技术。它通过在每个批次的数据上进行归一化操作来稳定训练过程。
        虚拟批量大小用于控制归一化操作中的批次大小。
        这两个变量的设置通常需要根据模型的结构和训练数据的性质进行调整。
        '''

        # 设置随机种子
        torch.manual_seed(self.seed)
        # Defining device
        # 定义gpu
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            warnings.warn(f"Device used : {self.device}")

        # create deep copies of mutable parameters
        '''
        通过使用 copy.deepcopy() 函数来创建深层副本（deep copies）的可变参数。
        深层副本是一种复制对象的方式，它递归地复制对象及其嵌套对象，以确保对象的完全独立性。
        这对于避免在修改副本时影响原始对象非常有用，因为它们不共享内部数据结构。
        '''
        self.optimizer_fn = copy.deepcopy(self.optimizer_fn)
        self.scheduler_fn = copy.deepcopy(self.scheduler_fn)

        # 离散类别特征 更新三个embedding有关变量
        updated_params = check_embedding_parameters(self.cat_dims,
                                                    self.cat_idxs,
                                                    self.cat_emb_dim)
        self.cat_dims, self.cat_idxs, self.cat_emb_dim = updated_params

        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'
        """
        self._task = 'classification': 这里指定了模型的任务类型，即分类任务。
            这表明你的模型将用于分类问题，通常需要使用适当的损失函数和评估指标。
       
        self._default_loss = torch.nn.functional.cross_entropy: 这里设置了默认损失函数为交叉熵损失（Cross-Entropy Loss）。
        交叉熵损失是在分类任务中常用的损失函数，用于度量模型的预测与实际标签之间的差异。
        torch.nn.functional.cross_entropy 是 PyTorch 中的一个函数，用于计算交叉熵损失。
        
        self._default_metric = 'accuracy': 这里设置了默认的评估指标为准确度（accuracy）。
            在分类任务中，准确度通常用于度量模型的性能，表示模型正确分类的样本比例。
        """

    def __update__(self, **kwargs):
        """
        更新参数。
        如果参数不存在，则创建它。
        否则用警告覆盖。
        更新参数，主要是用于无监督预训练
        """
        update_list = [
            "cat_dims",
            "cat_emb_dim",
            "cat_idxs",
            "input_dim",
            "mask_type",
            "n_a",
            "n_d",
            "n_independent",
            "n_shared",
            "n_steps",
            "grouped_features",
        ]
        for var_name, value in kwargs.items():
            if var_name in update_list:
                try:
                    exec(f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # noqa
                        wrn_msg = f"Pretraining: {var_name} changed from {previous_val} to {value}"  # noqa
                        warnings.warn(wrn_msg)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

    def fit(
        self,
        # 训练数据的特征。通常是一个包含输入数据的矩阵或张量。
        X_train,
        # 训练数据的标签。这是模型应该预测的目标变量或输出。
        y_train,
        # 用于验证的数据集。通常包括验证集的特征和标签。
        eval_set=None,
        # 用于验证的数据集的名称。
        eval_name=None,
        # 用于评估验证结果的指标，分为AUC和ACC。
        # AUC表示ROC曲线下的面积。AUC的取值范围通常在0.5到1之间，值越接近1表示模型性能越好。
        # ACC是另一种常用的分类性能指标，它表示模型正确预测的样本数量占总样本数量的比例。ACC的取值范围在0到1之间，值越接近1表示模型性能越好。
        eval_metric=None,
        # 损失函数，用于度量模型预测与真实标签之间的差异。
        loss_fn=None,
        # 样本权重，用于调整不同样本的重要性。
        weights=0,
        #  最大训练周期数，指定训练的最大迭代次数。
        max_epochs=100,
        # 用于早停（early stopping）的参数，表示如果验证指标不再改善，可以忍受的迭代次数。
        patience=10,
        # 批量大小，每个训练迭代中用于更新模型参数的样本数量。
        batch_size=1024,
        # 虚拟批量大小，通常与批规范化相关，表示批规范化操作中的虚拟批量大小。
        # 在批规范化中，统计信息（均值和方差）是在每个批次中计算的。
        # 虚拟批量大小是指在计算批规范化时使用的样本数量，这个数量可以大于实际批量大小。
        virtual_batch_size=128,
        # 数据加载器的工作线程数，用于加速数据加载
        num_workers=1,
        #  是否丢弃最后一个批次，通常用于确保所有批次都具有相同的大小。
        drop_last=True,
        # 训练回调函数的列表，用于在训练过程中执行额外的操作,如模型保存、学习率调整等。
        callbacks=None,
        # 是否将数据加载到 GPU 中的固定内存中，以加速训练。
        pin_memory=True,
        # 是否从无监督训练中加载参数。
        from_unsupervised=None,
        # 是否启用热启动，即从之前训练的模型状态开始训练。
        warm_start=False,
        # 数据增强操作，用于增加训练数据的多样性。
        augmentations=None,
        # 是否计算特征重要性。
        compute_importance=True
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.
        训练一个存储在 self.network 中的神经网络，
        使用 train_dataloader 作为训练数据，
        valid_dataloader 作为验证数据。、
        """
        # update model name

        self.max_epochs = max_epochs # 最大训练周期
        self.patience = patience # 是否早停
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers # 数据加载器工作线程数
        self.drop_last = drop_last # 丢弃最后一个批次
        self.input_dim = X_train.shape[1] # 输入数据的维度
        self._stop_training = False # 表示是否停止训练的状态
        self.pin_memory = pin_memory and (self.device.type != "cpu") # 是否将数据加载到 GPU 中的固定内存中
        # 数据增广方法
        self.augmentations = augmentations # 数据增强
        self.compute_importance = compute_importance # 计算特征重要性

        # 如果使用SMOTE等数据增广方法，为了保证可复现性，需要设定种子
        if self.augmentations is not None:
            # This ensure reproducibility
            self.augmentations._set_seed()

        # 设定需要验证的样本集合
        eval_set = eval_set if eval_set else []

        # 设置loss函数，默认为交叉熵
        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        # 检查输入规范
        check_input(X_train)
        # 检查是否热启动
        check_warm_start(warm_start, from_unsupervised)

        # 初始化输入输出，需要验证的集合以及类别或样本
        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        # 检查每个集合是否符合规范
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        # 生成两个dataloader，注意：valid_dataloaders是list
        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )

        # 若是热启动，则使用无监督训练好的参数
        if from_unsupervised is not None:
            # Update parameters to match self pretraining
            self.__update__(**from_unsupervised.get_params())
        # 如果不是热启动，则初始化一个主干网络
        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()
        # 更新网络参数
        self._update_network_params()
        # 设置每个集合的指标
        self._set_metrics(eval_metric, eval_names)
        # 设置优化器
        self._set_optimizer()
        # 设置回调函数
        self._set_callbacks(callbacks)
        # 加载参数
        if from_unsupervised is not None:
            self.load_weights_from_unsupervised(from_unsupervised)
            warnings.warn("Loading weights from unsupervised pretraining")
        # Call method on_train_begin for all callbacks
        # 训练开始触发回调
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs): # 最大训练周期

            # Call method on_epoch_begin for all callbacks
            # epoch开始触发回调
            self._callback_container.on_epoch_begin(epoch_idx)

            # 单个epoch训练
            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            # 所有集合进行预测
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            # epoch结束回调，此时epoch_metrics字典里有>3个键值：loss、学习率和不同集合的准确率、auc等指标
            # 打印所有log
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )
            # 看是否早停
            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        # 训练结束回调
        self._callback_container.on_train_end()
        # 进入eval模式
        self.network.eval()
        # 计算特征重要性
        if self.compute_importance:
            # compute feature importance once the best model is defined
            self.feature_importances_ = self._compute_feature_importances(X_train)




    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        # 这个函数方便之处在于可适配任意分类数据集
         # 最终输出维度就是类标签数量，softmax # 获得每个类标签
        output_dim, train_labels = infer_output_dim(y_train)
        # 检查label是否一致
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        # 初始化输出维度
        self.output_dim = output_dim
        # 设置评价指标为auc，若是二分类则为acc
        # ACC是另一种常用的分类性能指标，它表示模型正确预测的样本数量占总样本数量的比例
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        # 所有类别
        self.classes_ = train_labels
        # 初始化类别映射，str:int
        # 样的映射通常在训练过程中用于处理类别标签，将其转化为模型可处理的形式（整数形式）。
        # 这在训练中可能有助于对类别进行编码和处理。
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        # 初始化预测映射，上面那个反一下
        # 将整数映射回相应的类别。这在模型预测的输出中使用，将模型输出的整数形式的预测结果映射回原始的类别标签。
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        # 获取权重初始化变量，int or dict or list
        self.updated_weights = self.weight_updater(weights)
    
    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.
        类别权重更新函数

        通过使用类别权重，
        你可以赋予不同类别不同的重要性，使模型更加关注样本数量较少的类别，
        从而提高模型对整体数据集的泛化能力。

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        # 如果是0或1那么直接返回
        if isinstance(weights, int):
            return weights
        # 如果是dict，那么就是建个字典保存一下
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        # 如果是长度为训练集样本总量的list，那么也直接返回
        else:
            return weights
    
    def _construct_loaders(self, X_train, y_train, eval_set):
        """为训练集和验证集生成数据加载器。
        构造数据加载器

        Parameters
        ----------
        X_train : np.array
            训练集。
        y_train : np.array
            训练目标。
        eval_set : list of tuple
            评估元组集合列表（X，y）。

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            训练数据加载器。
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            验证数据加载器列表。
        """
        # all weights are not allowed for this type of model
        # 获取标签映射，str->int
        y_train_mapped = self.prepare_target(y_train)
        # 对每个集合同样操作
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        # 生成dataloader
        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders
    
    def prepare_target(self, y):
        # 函数向量化，可以看作是对y里面的每个值进行dict的get，来得到对应的类别编码
        return np.vectorize(self.target_mapper.get)(y)
    
    def _set_network(self):
        """Setup the network and explain matrix."""
        # 设置随机种子
        torch.manual_seed(self.seed)
        # 创建group权重矩阵
        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        # 初始化主干网络
        self.network = TabNet.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix( # 这个函数用来创建一个矩阵，该矩阵用于解释模型输出。
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def _update_network_params(self):
        # 设置GBN的BS
        # 打印VBS大小
        print(f"virtual_batch_size: {self.network.virtual_batch_size}")
        self.network.virtual_batch_size = self.virtual_batch_size
    
    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        # 为每个集合设置指标容器
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        # 设置Early stopping的指标
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(
            self.network.parameters(), **self.optimizer_params
        )
    
    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        # 初始化历史信息回调输出
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        # 如果有early stopping，就在list里加early stopping的回调，
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=(
                    self._metrics[-1]._maximize if len(self._metrics) > 0 else None
                ),
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            wrn_msg = "No early stopping will be performed, last training weights will be used."
            warnings.warn(wrn_msg)

        # 设置训练调度器
        if self.scheduler_fn is not None:
            # Add LR Scheduler call_back
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)
        # 添加自己设定的回调函数
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        # 把回调函数list装到容器里，方便触发
        self._callback_container = CallbackContainer(callbacks)
        # 把当前模型类实例赋值到容器里，意味着所有操作围绕TableClassifier展开，包括变量赋值、修改
        self._callback_container.set_trainer(self)

    def load_weights_from_unsupervised(self, unsupervised_model):
        # 从无监督预训练的模型里抽取模型参数并复制到当前实例化的模型
        update_state_dict = copy.deepcopy(self.network.state_dict())
        for param, weights in unsupervised_model.network.state_dict().items():
            if param.startswith("encoder"):
                # Convert encoder's layers name to match
                new_param = "tabnet." + param
            else:
                new_param = param
            if self.network.state_dict().get(new_param) is not None:
                # update only common layers
                update_state_dict[new_param] = weights

        self.network.load_state_dict(update_state_dict)

    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network
        单个epoch训练流程

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        # 进入train模式
        self.network.train()
        # 正常迭代训练
        for batch_idx, (X, y) in enumerate(train_loader):

            # batch开始回调
            self._callback_container.on_batch_begin(batch_idx)
            # 返回batch的log
            batch_logs = self._train_batch(X, y)
            # batch结束回调
            self._callback_container.on_batch_end(batch_idx, batch_logs)
        # 记录当前epoch的学习率
        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        # 此时epoch_metrics字典里有1个键值：学习率
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data
        单个batch训练流程

        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        # 记录bs
        batch_logs = {"batch_size": X.shape[0]}

        # 把x和y放到cpu或gpu里面
        X = X.to(self.device).float()
        y = y.to(self.device).float()
        # # # 打印数据和模型所在的设备
        # print(f"Data (X) device: {X.device}, Labels (y) device: {y.device}")
        # 增广
        if self.augmentations is not None:
            X, y = self.augmentations(X, y)
        # 初始化梯度
        for param in self.network.parameters():
            param.grad = None
        # 调用forward，返回输出和稀疏损失
        output, M_loss = self.network(X)

        # 计算分类损失
        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        # 计算整体损失
        loss = loss - self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        # 反向传播
        loss.backward()
        # 梯度裁剪
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        # 调用优化器
        self._optimizer.step()
        # 记录损失，此时batch_logs字典里有两个键值：batch_size和loss
        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def compute_loss(self, y_pred, y_true):
        # 返回损失
        return self.loss_fn(y_pred, y_true.long())
    
    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        # 进入eval模式
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)
        # 处理输出
        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        # 记录每个集合的指标
        metrics_logs = self._metric_container_dict[name](y_true, scores)
        # 再次进入训练模式
        self.network.train()
        # 此时epoch_metrics字典里有>2个键值：学习率和不同集合的准确率、auc等指标
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.
        batch层面的预测

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        # 放入gpu或cpu中
        X = X.to(self.device).float()

        # compute model output
        # 获得输出
        scores, _ = self.network(X)

        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()

        return scores

    def stack_batches(self, list_y_true, list_y_score):
        # 合并所有batch的输出和真实标签
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        # 输出softmax分数
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def _compute_feature_importances(self, X):
        """Compute global feature importance.
        计算全局特征重要性

        Parameters
        ----------
        loader : `torch.utils.data.Dataloader`
            Pytorch dataloader.

        """
        # 获得全局可解释性
        M_explain, _ = self.explain(X, normalize=False)
        # 所有样本累加
        sum_explain = M_explain.sum(axis=0)
        # 计算重要性
        feature_importances_ = sum_explain / np.sum(sum_explain)
        return feature_importances_

    def explain(self, X, normalize=False):
        """
        Return local explanation

        Parameters
        ----------
        X : tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data
        normalize : bool (default False)
            Wheter to normalize so that sum of features are equal to 1

        Returns
        -------
        M_explain : matrix
            Importance per sample, per columns.
        masks : matrix
            Sparse matrix showing attention masks used by network.
        """
        # 进入eval模式
        self.network.eval()
        # 构造预测数据集，即没有shuffle，数据增强，权重采样等操作
        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        res_explain = []

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            # 获得全局可解释性和每一个step的mask
            M_explain, masks = self.network.forward_masks(data)
            # 根据embedding后的矩阵进行dot操作，相当于把所有样本的每一个特征进行累加
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )
            # 同样操作
            original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(),
                                                   self.reducing_matrix)
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        # 归一化
        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, res_masks

    def save_model(self, path):
        """Saving TabNet model in two distinct files.

        Parameters
        ----------
        path : str
            Path of the model.

        Returns
        -------
        str
            input filepath with ".zip" appended

        """
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath):
        """Load TabNet model.

        Parameters
        ----------
        filepath : str
            Path of the model.
        """
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params["init_params"])

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        self.load_class_attrs(loaded_params["class_attrs"])

        return

    def load_class_attrs(self, class_attrs):
        for attr_name, attr_value in class_attrs.items():
            setattr(self, attr_name, attr_value)

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res
