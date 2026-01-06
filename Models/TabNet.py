import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from Models.FeatureTrans import FeatTransformer
from Models.AttentionTrans import AttentiveTransformer

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class TabNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=[],
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            初始特征维度
        output_dim : int
            网络输出的维度
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            预测层的维度
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Attention层的维度
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            网络的整体step数
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            分类特征的所属index
            Index of each categorical column in the dataset
        cat_dims : list of int
            分类特征的维度
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            embedding后的分类特征维度
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            每个特征的初始重要性权重矩阵
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        # 检查超参数（认为设定的参数）设置是否合理
        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim # post_embed_dim是embedding之后的特征维度

        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
            self.embedder.embedding_group_matrix
        )

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)

class EmbeddingGenerator(torch.nn.Module):
    """
    分类变量embedding生成(可选)
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            输入的特征维度
            Number of features coming as input (number of columns)
        cat_dims : list of int
            每个分类特征取值个数
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            每个分类特征的index
            Positional index for each categorical features in inputs
        cat_emb_dims : list of int
            每个分类特征的embedding维度
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features

        cat_dims（Category Dimensions）：是一个列表，用于指定每个分类特征的类别数量。
        例如，如果有两个分类特征，第一个有5个类别，第二个有3个类别，那么cat_dims可以设置为[5, 3]。

        cat_idxs（Category Indexes）：是一个列表，用于指定数据中哪些列是分类特征。
        列表中的每个元素是数据中对应分类特征的列索引。例如，如果数据的第2列和第5列是分类特征，
        那么cat_idxs可以设置为[2, 5]。

        cat_emb_dims（Category Embedding Dimensions）：是一个整数或整数列表，
        用于指定对每个分类特征进行嵌入（embedding）后的维度。
        如果是整数，表示所有分类特征的嵌入维度相同；如果是整数列表，表示每个分类特征的嵌入维度可以不同。

        这些参数的作用是为了在模型中处理分类特征。
        在TabNet中，分类特征会经过嵌入层（Embedding Layer）进行嵌入，
        嵌入后的特征会与连续特征一起输入到TabNet的结构中进行训练。
        这有助于模型更好地学习和处理具有分类特征的数据。
        嵌入（Embedding）是一种将离散型数据（如分类特征中的类别）映射到连续型空间的技术。
        嵌入层（Embedding Layer）用于学习这种映射关系，将高维的离散型数据转换为低维的连续型表示，从而提供了更好的特征表示。

        group_matrix : torch matrix
            原始group矩阵
            Original group matrix before embeddings
        """
        super(EmbeddingGenerator, self).__init__()

        # 如果不做embedding就跳过，并返回
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.embedding_group_matrix = group_matrix.to(group_matrix.device)
            return
        else:
            self.skip_embedding = False

        # 计算embedding之后的特征维度
        # 初始特征维度 input_dim 与所有分类变量嵌入的维度相加，然后减去分类变量的数量
        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        # 新建一个list储存每个embedding层
        self.embeddings = torch.nn.ModuleList()

        # 初始化每个embedding层
        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))
        # cat_dim 指定了嵌入层的输入维度，而 emb_dim 指定了嵌入层的输出维度

        # 记录连续变量的index，并设为0，分类变量设为1
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

        # 初始化新的group矩阵，大小为(n_groups, post_embed_dim)
        n_groups = group_matrix.shape[0] # 特征权重矩阵
        self.embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim),
                                                  device=group_matrix.device)
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # 如果是连续变量就不做embedding
                    self.embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]  # noqa
                    post_emb_idx += 1
                else:
                    # 做embedding并存到新的group矩阵中
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    self.embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings  # noqa
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

    def forward(self, x):
        """
        对每个样本做embedding
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            特征数量
            Number of features
        output_dim : int or list of int for multi task classification
            网络输出的维度
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            预测层的维度
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Attention层的维度
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            用于调整 Attention 更新的缩放因子。
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            所有批次归一化中使用的动量
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01) # 批量归一化

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False) # 通过 final_mapping 将 decision_features 映射到 output_dim 维度。
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        # 把每一个step的输出相加得到最终输出
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            # 最后一个全连接层
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)

class TabNetEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        定义Encoder

        Parameters
        ----------
        input_dim : int
            特征数
            Number of features
        output_dim : int or list of int for multi task classification
            输出维度
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            预测层的维度
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Attention层的维度
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            步数,for循环多少次
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            独立的GLU层个数
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            共享的GLU层个数
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            group注意力权重矩阵
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01) # 归一化
        self.group_attention_matrix = group_attention_matrix

        if self.group_attention_matrix is None:
            # 若group注意力矩阵为空，则初始化一个矩阵，每个特征单独为一个group，始终保持attention_dim = group数
            self.group_attention_matrix = torch.eye(self.input_dim)
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]

        if self.n_shared > 0: # 共享部分
            # 循环放入共享层中的FC层，注意：一般第一个共享层同样为整个FeatureTrans的第一层，输入维度为input_dim
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        # 输出维度始终要*2，因为GLU层的特性，实际最终GLU的最终输出维度仍然是一倍
                        Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        else:
            shared_feat_transform = None

        # 最开始的Split模块，首先是一个FeatureTrans，注意：输出维度为n_d + n_a
        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        # 初始化两个装featuretrans和attentiontrans层的list
        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        # 循环装入featuretrans和attentiontrans，注意：attentiontrans的输入维度为n_a
        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.attention_dim,
                group_matrix=group_attention_matrix, # 18*18
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
                mask_type=self.mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        # 先是一个BN，做归一化
        x = self.initial_bn(x)

        bs = x.shape[0]  # batch size
        # 定义batch中的每个样本对每个group的先验注意力权重
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        M_loss = 0
        # 首先得到split后的第一个注意力矩阵，维度：（batch_size, n_a）
        att = self.initial_splitter(x)[:, self.n_d :]
        steps_output = []
        for step in range(self.n_steps):
            # 循环输入attentiontrans中，并得到经sparsemax后的矩阵，维度为：（batch_size, attention_dim）
            M = self.att_transformers[step](prior, att)
            # 稀疏性损失计算，进一步控制特征选择稀疏性
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            # 更新先验权重
            prior = torch.mul(self.gamma - M, prior)

            # 根据group注意力矩阵和每个group的权重，计算得到每个特征在每个样本上的权重
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            # 接着将计算得到的权重乘到输入上
            masked_x = torch.mul(M_feature_level, x)

            # 输入到featuretrans中
            out = self.feat_transformers[step](masked_x)
            # 把预测层输出输入到ReLU中，激活函数，维度：（batch_size, n_d）
            d = ReLU()(out[:, : self.n_d])
            # 记录每一步的预测层输出
            steps_output.append(d)
            # 更新attention矩阵，维度为：（batch_size, n_a）
            att = out[:, self.n_d :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)
        bs = x.shape[0]  # batch size
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        # 初始化explain矩阵
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            # M可以理解为每个group的重要性矩阵
            M = self.att_transformers[step](prior, att)
            # 得到每个step的特征重要性矩阵
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            masks[step] = M_feature_level
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            # explain
            # 获得可解释性
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks