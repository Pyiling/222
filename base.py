import torch
import torch.nn as nn
import numpy as np


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GLULayer(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None):
        super(GLULayer, self).__init__()
        self.fc = fc if fc else nn.Linear(input_dim, output_dim * 2)
        initialize_glu(self.fc, input_dim, output_dim)

    def forward(self, x):
        x_proj = self.fc(x)
        output, gate = x_proj.chunk(2, dim=-1)
        return output * torch.sigmoid(gate)


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers=None, n_glu=2):
        super(FeatureTransformer, self).__init__()
        self.glu_layers = nn.ModuleList()

        for glu_id in range(n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLULayer(input_dim if glu_id == 0 else output_dim, output_dim, fc=fc))

    def forward(self, x):
        for layer in self.glu_layers:
            x = layer(x)
        return x


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, n_d, n_a, shared_layers=None, n_glu=2):
        super(TabNetEncoder, self).__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.feature_transformer = FeatureTransformer(input_dim, n_d + n_a, shared_layers, n_glu)

    def forward(self, x):
        transformed_features = self.feature_transformer(x)
        decision_features, attention_features = torch.split(transformed_features, [self.n_d, self.n_a], dim=-1)
        return decision_features, attention_features


# 示例参数
input_dim = 16
n_d = 8
n_a = 8
shared_layers = [nn.Linear(input_dim, (n_d + n_a) * 2) for _ in range(2)]

# 创建TabNet编码器
tabnet_encoder = TabNetEncoder(input_dim, n_d, n_a, shared_layers)

# 示例输入
x = torch.randn(8, input_dim)

# 前向传播
decision_features, attention_features = tabnet_encoder.forward(x)
print("Decision Features Shape:", decision_features.shape)  # (8, 8)
print("Attention Features Shape:", attention_features.shape)  # (8, 8)
