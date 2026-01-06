import argparse  # 命令行选项、参数和子命令解析器
import torch
from matplotlib import pyplot as plt
# sklearn中常用的模块有分类、回归、聚类、降维、模型选择、预处理。
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd  # 数据变换
import numpy as np

np.random.seed(0)
import scipy
import os
from pathlib import Path
import sys

# from torchviz import make_dot


sys.path.append('/workspace/Models')
sys.path.append('/workspace/Utils')
from Models.TabNetClassification import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 特征归一化
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from Utils.Augmentations import ClassificationSMOTE
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('module://matplotlib_inline.backend_inline')
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve
from sklearn.metrics import auc as Auc
import scipy.stats as st
from sklearn import metrics

'''
1、创建一个解析器——创建 ArgumentParser() 对象
2、添加参数——调用 add_argument() 方法添加参数
3、解析参数——使用 parse_args() 解析添加的参数
'''
# 用来装载参数的容器
parser = argparse.ArgumentParser(description='TabNet')
# 初始化参数
parser.add_argument('--n_d', type=int, default=8, help='决策层维度')
parser.add_argument('--n_a', type=int, default=8, help='注意力层维度')
parser.add_argument('--n_steps', type=int, default=4, help='step数')
parser.add_argument('--gamma', type=float, default=1.0, help='prior更新权重')
parser.add_argument('--cat_idxs', type=list[int], default=[], help='类别特征的index列表')
parser.add_argument('--cat_dims', type=list[int], default=[], help='类别特征的特征取值数列表')
parser.add_argument('--cat_emb_dim', type=int, default=1, help='类别特征embedding后的维度')
parser.add_argument('--n_independent', type=int, default=3, help='独立层数')
parser.add_argument('--n_shared', type=int, default=3, help='共享层数')
parser.add_argument('--epsilon', type=float, default=1.5e-4, help='防止出现log(0)')
parser.add_argument('--momentum', type=float, default=0.09, help='BN层滑动平均系数')
parser.add_argument('--lambda_sparse', type=float, default=0.001, help='稀疏损失系数')

parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--clip_value', type=int, default=1, help='是否进行梯度裁剪')
parser.add_argument('--verbose', type=int, default=1, help='是否输出训练信息')
parser.add_argument('--optimizer_fn', type=any, default=torch.optim.Adam, help='优化器')
parser.add_argument('--optimizer_params', type=dict, default=dict(lr=0.0001), help='优化器参数字典')
parser.add_argument('--scheduler_fn', type=any, default=torch.optim.lr_scheduler.StepLR, help='调度器')
parser.add_argument('--scheduler_params', type=dict, default={"step_size": 50, "gamma": 0.9}, help='调度器参数字典')
parser.add_argument('--mask_type', type=str, default="sparsemax", help='稀疏性函数类型')
# sparsemax
parser.add_argument('--input_dim', type=int, default=None, help='输入维度')
parser.add_argument('--output_dim', type=int, default=None, help='输出维度')
parser.add_argument('--device_name', type=str, default="auto", help='运行设备')
parser.add_argument('--grouped_features', type=list[list[int]], default=[], help='group列表')

# 训练参数
parser_train = argparse.ArgumentParser(description='Train')
parser_train.add_argument('--X_train', type=np.array, default=None, help='训练集样本')
parser_train.add_argument('--y_train', type=np.array, default=None, help='训练集标签')
parser_train.add_argument('--eval_set', type=list, default=None, help='验证集合')
parser_train.add_argument('--eval_name', type=list, default=None, help='验证集合名')
parser_train.add_argument('--eval_metric', type=list, default=['accuracy','auc'], help='验证集合指标')
parser_train.add_argument('--loss_fn', type=any, default=None, help='损失函数')
parser_train.add_argument('--weights', type=any, default=0, help='样本权重')
parser_train.add_argument('--max_epochs', type=int, default=512, help='最大epoch数')
parser_train.add_argument('--patience', type=int, default=50, help='早停条件')
parser_train.add_argument('--batch_size', type=int, default=32, help='batch大小')
parser_train.add_argument('--virtual_batch_size', type=int, default=32, help='GBN batch大小')
parser_train.add_argument('--num_workers', type=int, default=0, help='设定自主加载数据到RAM')
parser_train.add_argument('--drop_last', type=bool, default=False, help='是否丢弃dataloader最后那一点数据')
parser_train.add_argument('--callbacks', type=list, default=None, help='回调函数')
parser_train.add_argument('--pin_memory', type=bool, default=True, help='是否pin memory') # 将数据加载到固定的（锁页的）内存中，这样可以加快从 CPU 到 GPU 的数据传输。
parser_train.add_argument('--from_unsupervised', type=any, default=None, help='预训练好的模型')
parser_train.add_argument('--warm_start', type=bool, default=False, help='是否热启动')
parser_train.add_argument('--augmentations', type=any, default=ClassificationSMOTE(p=0.2), help='数据增广')
parser_train.add_argument('--compute_importance', type=bool, default=True, help='是否计算特征重要性')
# ClassificationSMOTE(p=0.2)
# 测试参数
parser_test = argparse.ArgumentParser(description='Test')
parser_test.add_argument('--data_path', type=str, default='axspanew.csv', help='数据路径')

parser_test.add_argument('--data_test_path', type=str, default=r'D:\python\git\TableNet-main\test_clin.csv', help='数据路径')
parser_test.add_argument('--data_train_path', type=str, default=r'D:\python\git\TableNet-main\train_clin.csv', help='数据路径')
parser_test.add_argument('--save_path', type=str, default=r'D:\python\git\TableNet-main\model_file\diabetes\tabnet_model_test_5', help='模型存放路径')




def preprocessing_all(args, args_train, args_test):
    """
    数据预处理
    """
    # 读取数据
    X_train = pd.read_csv(args_test.data_train_path).values[:, :-1]
    y_train = pd.read_csv(args_test.data_train_path).values[:, -1]

    X_valid = pd.read_csv(args_test.data_test_path).values[:, :-1]
    y_valid = pd.read_csv(args_test.data_test_path).values[:, -1]

    # X = pd.read_csv(args_test.data_path).values[:, :-1]
    # y = pd.read_csv(args_test.data_path).values[:, -1]
    #
    # y = np.array(y, dtype=float)



    # 拆分数据集，训练集：验证集：测试集=1：1：1
    # X_train, X_valid_all, y_train, y_valid_all = train_test_split(X, y, train_size=0.8, stratify=y, random_state=0)

    # X_valid, X_test, y_valid, y_test = train_test_split(X_valid_all, y_valid_all, train_size=0.5, stratify=y_valid_all,
    #                                                     random_state=0)

    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

    print(X_train.shape)
    print(X_valid.shape)

    # 初始化标准化类
    scaler = StandardScaler()
    # # 训练集标准化，主要是处理均值和方差
    X_train = scaler.fit_transform(X_train)

    # 把训练集标准化参数应用到验证集和测试集上，不用再次fit防止数据泄露
    X_valid = scaler.transform(X_valid)

    # X_test = scaler.transform(X_test)

    # # 初始化标准化类
    # scaler = StandardScaler()
    #
    # # 训练集标准化，仅对第5列和第6列进行操作
    # X_train[:, [4, 5]] = scaler.fit_transform(X_train[:, [4, 5]])
    #
    # # 使用训练集的标准化参数应用到验证集上，仅对第5列和第6列进行操作
    # X_valid[:, [4, 5]] = scaler.transform(X_valid[:, [4, 5]])



    # 放到namespace里面
    args_train.X_train = X_train

    args_train.y_train = y_train

    args_train.eval_set = [(X_train, y_train), (X_valid, y_valid)]

    args_train.eval_name = ['train', 'valid']

    # 转字典，得到namespace里面所有键值对
    dict = vars(args)
    dict_train = vars(args_train)

    return dict, dict_train, X_train, X_valid, y_train, y_valid


def train(dict, dict_train, X_train, X_test, y_train, y_test, args_test):
    """
    普通训练
    """
    # 得到所有特征名字
    features = pd.read_csv(args_test.data_train_path).columns

    clf = TabNetClassifier(**dict)

    clf.fit(**dict_train)

    # 保存模型
    saved_filepath = clf.save_model(args_test.save_path)

    # 读取保存的模型参数
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(saved_filepath)

    # 测试集预测，注意：predict_proba函数只能得到每个样本在每个类上的softmax概率，可以直接算auc，算准确率要转换一下
    loaded_preds = loaded_clf.predict_proba(X_test)

    # 算auc
    auc = roc_auc_score(y_score=loaded_preds[:,1], y_true=y_test)


    # 算acc
    acc = accuracy_score(y_pred=loaded_preds.argmax(1), y_true=y_test)

    # 计算 Recall
    # recall = recall_score(y_true=y_test, y_pred=loaded_preds.argmax(1))

    # 计算 Precision
    precision = precision_score(y_true=y_test, y_pred=loaded_preds.argmax(1))

    # 计算sensitivity以及specificity

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=loaded_preds.argmax(1))

    # 获取混淆矩阵的元素
    tn, fp, fn, tp = conf_matrix.ravel()

    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 计算fs
    fs = f1_score(y_true=y_test, y_pred=loaded_preds.argmax(1))

    # 计算mcc
    mc = matthews_corrcoef(y_true=y_test, y_pred=loaded_preds.argmax(1))

    y_pred_proba = loaded_preds[:, 1]

    n_bootstraps = 1000
    auc_bootstrap = []
    accuracy_bootstrap = []
    specificity_bootstrap = []
    sensitivity_bootstrap = []
    f1_bootstrap = []
    mcc_bootstrap = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_bootstrap = y_test[indices]
        y_pred_proba_bootstrap = y_pred_proba[indices]
        y_pred_bootstrap = (y_pred_proba_bootstrap > 0.5).astype(int)

        auc_bootstrap.append(roc_auc_score(y_test_bootstrap, y_pred_proba_bootstrap))
        accuracy_bootstrap.append(accuracy_score(y_test_bootstrap, y_pred_bootstrap))
        tn, fp, fn, tp = confusion_matrix(y_test_bootstrap, y_pred_bootstrap).ravel()
        specificity_bootstrap.append(tn / (tn + fp))
        sensitivity_bootstrap.append(tp / (tp + fn))
        f1_bootstrap.append(f1_score(y_test_bootstrap, y_pred_bootstrap))
        mcc_bootstrap.append(matthews_corrcoef(y_test_bootstrap, y_pred_bootstrap))

    # 计算95%置信区间
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    auc_ci = np.percentile(auc_bootstrap, [lower_percentile, upper_percentile])
    accuracy_ci = np.percentile(accuracy_bootstrap, [lower_percentile, upper_percentile])
    specificity_ci = np.percentile(specificity_bootstrap, [lower_percentile, upper_percentile])
    sensitivity_ci = np.percentile(sensitivity_bootstrap, [lower_percentile, upper_percentile])
    f1_ci = np.percentile(f1_bootstrap, [lower_percentile, upper_percentile])
    mcc_ci = np.percentile(mcc_bootstrap, [lower_percentile, upper_percentile])

    # print("Accuracy 95% Confidence Interval: ({:.2f},{:.2f})".format(*accuracy_ci))
    # print("AUC 95% Confidence Interval: ({:.2f},{:.2f})".format(*auc_ci))
    #
    # print("Sensitivity 95% Confidence Interval: ({:.2f},{:.2f})".format(*sensitivity_ci))
    # print("Specificity 95% Confidence Interval: ({:.2f},{:.2f})".format(*specificity_ci))
    # print("F1 Score 95% Confidence Interval: ({:.2f},{:.2f})".format(*f1_ci))
    # print("MCC 95% Confidence Interval: ({:.2f},{:.2f})".format(*mcc_ci))

    # print("preds_A", loaded_preds[:,1])

    # # 创建一个DataFrame保存新的结果
    # result_df = pd.DataFrame({
    #     'Model': ['TabNet'],
    #     'Accuracy': [acc],
    #     'AUC': [auc],
    #     # 'Recall': [recall],
    #     # 'Precision': [precision],
    #     'sensitivity': [sensitivity],
    #     'specificity': [specificity],
    #     'F1 Score': [fs],
    #     'MCC': [mc]
    # })
    # existing_results_path = r'D:\github_python\TableNet-main\all_Test.csv'
    # # 如果已有的CSV文件存在，则追加新的结果
    # if os.path.exists(existing_results_path):
    #     existing_results_df = pd.read_csv(existing_results_path)
    #     combined_df = pd.concat([existing_results_df, result_df], ignore_index=True)
    #     combined_df.to_csv(existing_results_path, index=False)
    # else:
    #     # 如果已有的CSV文件不存在，则创建一个新的文件并保存新的结果
    #     result_df.to_csv(existing_results_path, index=False)
    # print(f"acc: {acc}, auc: {auc},sensitivity: {sensitivity},specificity: {specificity},f-measure: {fs},MCC: {mc}")
    print(f"acc: {acc:.2%}({accuracy_ci[0]:.2f},{accuracy_ci[1]:.2f}), auc: {auc:.2%}({auc_ci[0]:.2f},{auc_ci[1]:.2f}), sensitivity: {sensitivity:.2%}({sensitivity_ci[0]:.2f},{sensitivity_ci[1]:.2f}), specificity: {specificity:.2%}({specificity_ci[0]:.2f},{specificity_ci[1]:.2f}), f-measure: {fs:.2%}({f1_ci[0]:.2f},{f1_ci[1]:.2f}), MCC: {mc:.2%}({mcc_ci[0]:.2f},{mcc_ci[1]:.2f})")



    fpr, tpr, _ = roc_curve(y_test, loaded_preds[:,1])
    roc_auc = Auc(fpr, tpr)
    print(f"fpr:{fpr}")
    print(f"tpr:{tpr}")
    print(f"roc_auc:{roc_auc}")







    # loaded_preds1 = loaded_clf.predict_proba(X_train)
    # # 算auc
    # auc1 = roc_auc_score(y_score=loaded_preds1[:, 1], y_true=y_train)
    #
    # # 算acc
    # acc1 = accuracy_score(y_pred=loaded_preds1.argmax(1), y_true=y_train)
    #
    # # 计算 Recall
    # # recall1 = recall_score(y_true=y_train, y_pred=loaded_preds1.argmax(1))
    #
    # # 计算 Precision
    # precision1 = precision_score(y_true=y_train, y_pred=loaded_preds1.argmax(1))
    #
    # # 计算sensitivity以及specificity
    #
    # # 计算混淆矩阵
    # conf_matrix1 = confusion_matrix(y_true=y_train, y_pred=loaded_preds1.argmax(1))
    #
    # # 获取混淆矩阵的元素
    # tn, fp, fn, tp = conf_matrix1.ravel()
    #
    # # 计算敏感性和特异性
    # sensitivity1 = tp / (tp + fn)
    # specificity1 = tn / (tn + fp)
    #
    # # 计算fs
    # fs1 = f1_score(y_true=y_train, y_pred=loaded_preds1.argmax(1))
    #
    # # 计算mcc
    # mc1 = matthews_corrcoef(y_true=y_train, y_pred=loaded_preds1.argmax(1))
    #
    # # # 创建一个DataFrame保存新的结果
    # # result_df = pd.DataFrame({
    #     'Model': ['TabNet'],
    #     'Accuracy': [acc1],
    #     'AUC': [auc1],
    #     # 'Recall': [recall1],
    #     # 'Precision': [precision1],
    #     'sensitivity': [sensitivity1],
    #     'specificity': [specificity1],
    #     'F1 Score': [fs1],
    #     'MCC': [mc1]
    # })
    # existing_results_path = r'D:\github_python\TableNet-main\all_Train.csv'
    # # 如果已有的CSV文件存在，则追加新的结果
    # if os.path.exists(existing_results_path):
    #     existing_results_df = pd.read_csv(existing_results_path)
    #     combined_df = pd.concat([existing_results_df, result_df], ignore_index=True)
    #     combined_df.to_csv(existing_results_path, index=False)
    # else:
    #     # 如果已有的CSV文件不存在，则创建一个新的文件并保存新的结果
    #     result_df.to_csv(existing_results_path, index=False)

    # print(f"FINAL TEST SCORE FOR {'acc'} : {loaded_test_acc},auc:{loaded_test_auc}")
    # print(f"acc: {acc1}, auc: {auc1},sensitivity: {sensitivity1},specificity: {specificity1},f-measure: {fs1},MCC: {mc1}")

    # print(f"FEATURE IMPORTANCES FOR {'axpan'} : {clf.feature_importances_}")
    # # 已知的特征和对应的重要性
    # features_all = ['gender', 'age', 'duration', 'HLA-B27', 'ESR', 'CRP', 'erosion-right', 'erosion-left', 'sclerosis-right', 'sclerosis-left',
    #             'joint space-right', 'joint space-left', 'ankylosis-right', 'ankylosis-left', 'fat lesion-right', 'fat lesion-left', 'BME-right', 'BME-left']
    #
    # importance_values_all=[0.06316288, 0.07810093, 0.05173627, 0.09698079, 0.02616658, 0.05142958,
    #            0.05038133, 0.04780184, 0.02649785, 0.03335315, 0.11163425, 0.09357796,
    #            0.0330305, 0.03486708, 0.03977188, 0.03966361, 0.07534591, 0.04649763]
    #
    # # features_ct = ['erosion-right', 'erosion-left', 'sclerosis-right', 'sclerosis-left',
    # #             'joint space-right', 'joint space-left', 'bone hyperplasia-right', 'bone hyperplasia-left', 'fat deposits-right', 'fat deposits-left', 'BME-right', 'BME-left']
    # # importance_values_ct = [0.05634787, 0.06037921, 0.02749263, 0.11384201, 0.12569028, 0.12152051,
    # #                      0.03781051, 0.06239136, 0.1924051, 0.02832346, 0.04571719, 0.12807987]
    # #
    # # features_clin = ['gender', 'age', 'duration', 'HLA-B27', 'ESR', 'CRP']
    # # importance_values_clin = [0.13822992, 0.07072716, 0.11490858, 0.39805778, 0.10038559, 0.17769097]
    #
    # # 创建数据框
    # df = pd.DataFrame({'Features1': features_all, 'Importance1': importance_values_all})
    # # df = pd.DataFrame({'Features2': features_ct, 'Importance2': importance_values_ct})
    # # df = pd.DataFrame({'Features3': features_clin, 'Importance3': importance_values_clin})
    #
    # # 绘制柱状图
    # plt.figure(figsize=(10, 6))
    # plt.bar(df['Features1'], df['Importance1'], color='skyblue')
    # plt.xlabel('Features')
    # plt.ylabel('Importance Score')
    # plt.title('The feature importance of the combined clinical-imaging model.')
    # plt.xticks(rotation=45, ha='right')  # 使特征名字斜着显示，避免重叠
    # plt.tight_layout()  # 自动调整布局，避免标签被截断
    #
    # plt.show()

    # # 得到解释性矩阵和每个step的mask
    # explain_matrix, masks = clf.explain(X_test)
    #
    # fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    #
    # for i in range(3):
    #     axs[i].imshow(masks[i][:50])
    #     axs[i].set_title(f"mask {i}")
    #     axs[i].set_xticklabels(labels=features, rotation=45)

    # plt.show()





def train_5_fold(args, args_train, args_test):
    """
    五折交叉验证训练
    """
    # 读取
    X = pd.read_csv(args_test.data_path).values[:, 1:-1]  # 读入数据集
    y = pd.read_csv(args_test.data_path).values[:, -1]

    print(X.shape)  # 读取矩阵的长度

    y = np.array(y, dtype=float)
    '''
    numpy.array(object, dtype=None)
    object：创建的数组的对象，可以为单个值，列表，元胞等。
    dtype：创建数组中的数据类型。
    返回值：给定对象的数组。
    '''

    # 数据预处理
    # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
    scaler = StandardScaler()

    # 初始化K折类
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    '''
    StratifiedKFold函数采用分层划分的方法（分层随机抽样思想），
    验证集中不同类别占比与原始样本的比例保持一致，
    故StratifiedKFold在做划分的时候需要传入标签特征。

    class sklearn.model_selection.StratifiedKFold(n_splits=’warn’, shuffle=False, random_state=None)
    n_splits：表示几折（折叠的数量）在这里是分成五分数据
    shuffle== True:选择是否在分割成批次之前对数据的每个分层进行打乱。
                   供5次2折使用，这样每次的数据是进行打乱的，否则，每次取得的数据是相同的
    random_state:控制随机状态，随机数生成器使用的种子
    '''
    res = 0

    # 迭代五次
    # kf.split(x,y)返回的是数据集的索引，需要x[train_index] y[train_index]才能提取数据
    # train_index, test_index分别表示第几个数据是训练集或者测试集
    # X[train_index]X[test_index]y[train_index]y[test_index]表示提取训练集或测试集第几个位置的数据
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for j, (train_index, test_index) in enumerate(kf.split(X, y)):
        # 数据预处理
        X_train = scaler.fit_transform(X[train_index])  # 标准差标准化后的矩阵  归一化后的矩阵

        X_valid = scaler.transform(X[test_index])
        # 训练集计算出来的参数去缩放测试集 (即假设训练集和测试集来自同一个分布)，使得训练的模型更普适 (genelization)。
        '''
        Fit(): Method calculates the parameters μ and σ and saves them as internal objects.
        解释：简单来说，就是求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集X固有的属性。可以理解为一个训练过程

        Transform(): Method using these calculated parameters apply the transformation to a particular dataset.
        解释：在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。

        Fit_transform(): joins the fit() and transform() method for transformation of dataset.
        解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。

        transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）

        fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。

        根据对之前部分trainData进行fit的整体指标，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，从而保证train、test处理方式相同
        '''
        y_train = y[train_index]  # 取出训练集的标签

        y_valid = y[test_index]  # 取出验证集的标签

        args_train.X_train = X_train

        args_train.y_train = y_train

        args_train.eval_set = [(X_train, y_train), (X_valid, y_valid)]

        args_train.eval_name = ['train', 'valid']

        # 转字典，得到namespace里面所有键值对
        dict = vars(args)  # 把args按dict来解析
        dict_train = vars(args_train)

        # 初始化分类器
        clf = TabNetClassifier(**dict)  # 这通常用于将一个字典中的键值对传递给一个函数，而不需要逐个指定这些键值对的参数
        # 字典中的键必须与函数的参数名称匹配，否则会引发 TypeError。

        # 训练
        clf.fit(**dict_train)
        # best_loss是每次训练的最好的验证集准确率
        res += clf.best_loss

    res /= 5
    print(f"5 FOLD RESULTS FOR {'diabetes'} : {res}")



if __name__ == '__main__':
    print('1')
    args = parser.parse_args()  # 获取容器中的参数
    args_train = parser_train.parse_args()
    args_test = parser_test.parse_args()

    dict, dict_train, X_train, X_test, y_train, y_test = preprocessing_all(args, args_train, args_test)

    train(dict, dict_train, X_train, X_test, y_train, y_test, args_test)

    # train_5_fold(args, args_train, args_test)