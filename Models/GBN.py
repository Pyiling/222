import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    特殊的批归一化层,将batch处理为更小的batch,训练时做batch_size/virtual_batch_size次BN,
    测试时则用滑动平均后的全局均值和标准差,目的在于增加BN次数,核心认为越到后面BN获得的均值和标准差越接近整体
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        # 更小的Batch大小
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        # 多次BN并合并res
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)