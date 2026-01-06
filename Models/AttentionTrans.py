import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from Models.GLU import GLU_Block
import Models.sparsemax as sparsemax
from Models.GBN import GBN

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        group_matrix,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            输入维度
            Input size
        group_dim : int
            group的维度
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        # 首先一个fc层
        self.fc = Linear(input_dim, group_dim, bias=False)
        # 初始化非glu层参数
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GBN(
            group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):
        # 全连接层，shape:(bs, group_dim)
        x = self.fc(processed_feat)
        # BN层
        x = self.bn(x)
        # 先乘先验
        x = torch.mul(x, priors)
        # 稀疏化（根据group进行特征选择）
        x = self.selector(x)
        return x