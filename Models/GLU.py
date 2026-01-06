import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from Models.GBN import GBN

def initialize_glu(module, input_dim, output_dim):
    # 初始化GLU层参数
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        # 是否为第一个GLU_Block，一般共享权重的都是第一个
        self.first = first
        # 共享的GLU层
        self.shared_layers = shared_layers
        # 层数
        self.n_glu = n_glu
        # 初始化一个装module的list
        self.glu_layers = torch.nn.ModuleList()

        # 初始化两个超参数，并装到字典里
        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        # 共享层的第一个全连接层
        fc = shared_layers[0] if shared_layers else None
        # list里添加一个以这个fc为全连接层的GLU层
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        # 根据共享层里的fc层，循环生成GLU
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        # 缩放因子
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            # 如果是第一个block
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        # 把前一个GLU的res加到之后的GLU的res上，并缩放，跳跃连接
        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x

class GLU_Layer(torch.nn.Module):
    """
    GLU层,由一个全连接层加BN层
    """
    def __init__(
        self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        # GLU层的核心：在全连接层基础上再乘一个sigmoid后的全连接层，来控制输出，相当于对每个output的元素或者说样本进行了缩小，压缩信息
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        # 这里的BN层为GBN
        self.bn = GBN(
            2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        # 对每个位置进行缩放
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out