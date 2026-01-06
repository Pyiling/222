import argparse  # 命令行选项、参数和子命令解析器
import torch
import pandas as pd  # 数据变换
import numpy as np
np.random.seed(0)
import scipy
import os
from pathlib import Path
import sys
sys.path.append('/workspace/Models')
sys.path.append('/workspace/Utils')
from Models.TabNetClassification import TabNetClassifier
from sklearn.preprocessing import StandardScaler  # 特征归一化
from Utils.Augmentations import ClassificationSMOTE
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from PSO import pso_optimize, fitness_function

'''
1、创建一个解析器——创建 ArgumentParser() 对象
2、添加参数——调用 add_argument() 方法添加参数
3、解析参数——使用 parse_args() 解析添加的参数
'''
# 用来装载参数的容器
parser = argparse.ArgumentParser(description='TabNet')
# 初始化参数
parser.add_argument('--n_d', type=int, default=16, help='决策层维度')
parser.add_argument('--n_a', type=int, default=16, help='注意力层维度')
parser.add_argument('--n_steps', type=int, default=4, help='step数')
parser.add_argument('--gamma', type=float, default=1.5, help='prior更新权重')
parser.add_argument('--cat_idxs', type=list[int], default=[], help='类别特征的index列表')
parser.add_argument('--cat_dims', type=list[int], default=[], help='类别特征的特征取值数列表')
parser.add_argument('--cat_emb_dim', type=int, default=1, help='类别特征embedding后的维度')
parser.add_argument('--n_independent', type=int, default=3, help='独立层数')
parser.add_argument('--n_shared', type=int, default=3, help='共享层数')
parser.add_argument('--epsilon', type=float, default=1.5e-4, help='防止出现log(0)')
parser.add_argument('--momentum', type=float, default=0.09, help='BN层滑动平均系数')
parser.add_argument('--lambda_sparse', type=float, default=1e-4, help='稀疏损失系数')

parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--clip_value', type=int, default=1, help='是否进行梯度裁剪')
parser.add_argument('--verbose', type=int, default=1, help='是否输出训练信息')
parser.add_argument('--optimizer_fn', type=any, default=torch.optim.Adam, help='优化器')
parser.add_argument('--optimizer_params', type=dict, default=dict(lr=1.5e-3), help='优化器参数字典')
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
parser_train.add_argument('--max_epochs', type=int, default=600, help='最大epoch数')
parser_train.add_argument('--patience', type=int, default=100, help='早停条件')
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

parser_test.add_argument('--data_test_path', type=str, default=r'D:\python\git\TableNet-main\testnew_all_data.csv', help='数据路径')
parser_test.add_argument('--data_train_path', type=str, default=r'D:\python\git\TableNet-main\trainnew_all_data.csv', help='数据路径')
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
    print(X_train.shape)
    print(X_valid.shape)
    # 初始化标准化类
    scaler = StandardScaler()
    # # 训练集标准化，主要是处理均值和方差
    X_train = scaler.fit_transform(X_train)
    # 把训练集标准化参数应用到验证集和测试集上，不用再次fit防止数据泄露
    X_valid = scaler.transform(X_valid)
    # 放到namespace里面
    args_train.X_train = X_train

    args_train.y_train = y_train

    args_train.eval_set = [(X_train, y_train), (X_valid, y_valid)]

    args_train.eval_name = ['train', 'valid']

    # 转字典，得到namespace里面所有键值对
    dict = vars(args)
    dict_train = vars(args_train)

    return dict, dict_train, X_train, X_valid, y_train, y_valid

if __name__ == '__main__':
    print('1')
    args = parser.parse_args()  # 获取容器中的参数
    args_train = parser_train.parse_args()
    args_test = parser_test.parse_args()

    dict, dict_train, X_train, X_test, y_train, y_test = preprocessing_all(args, args_train, args_test)


    # 初始化记录表格
    results = []
    optimized_params_list = []
    for i in range(25):
        # 模型初始化
        model = TabNetClassifier(**dict)
        # 打印数据和模型所在的设备
        print(f"model device: {model.device}")
        # 使用 PSO 进行超参数优化
        best_params, best_score = pso_optimize(
            fitness_function,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            dict_train,
            n_particles=30,
            max_iters=50
        )
        print(f"最佳参数: {best_params}")
        print(f"最佳AUC: {best_score:.4f}")
        # 使用找到的最佳参数重新训练模型
        model.n_d, model.n_a, model.lambda_sparse, model.gamma, model.n_steps, \
            model.n_shared, args_train.max_epochs, args_train.batch_size, \
            args_train.virtual_batch_size, model.optimizer_params['lr'], \
            model.scheduler_params['step_size'] = best_params
        # 转换为整数
        args_train.max_epochs = int(args_train.max_epochs)
        args_train.batch_size = int(args_train.batch_size)
        args_train.virtual_batch_size = int(args_train.virtual_batch_size)
        model.n_d = int(model.n_d)
        model.n_a = int(model.n_a)
        model.n_steps = int(model.n_steps)
        model.n_shared = int(model.n_shared)

        # 输出优化后的参数
        print("Optimized Parameters:")
        print(f"n_d: {model.n_d}")
        print(f"n_a: {model.n_a}")
        print(f"lambda_sparse: {model.lambda_sparse}")
        print(f"gamma: {model.gamma}")
        print(f"n_steps: {model.n_steps}")
        print(f"n_shared: {model.n_shared}")
        print(f"max_epochs: {args_train.max_epochs}")
        print(f"batch_size: {args_train.batch_size}")
        print(f"virtual_batch_size: {args_train.virtual_batch_size}")
        print(f"learning rate (lr): {model.optimizer_params['lr']}")
        print(f"scheduler step size: {model.scheduler_params['step_size']}")
        # 将最优参数保存到列表中
        optimized_params_list.append({
            'iteration': i + 1,
            'n_d': model.n_d,
            'n_a': model.n_a,
            'lambda_sparse': model.lambda_sparse,
            'gamma': model.gamma,
            'n_steps': model.n_steps,
            'n_shared': model.n_shared,
            'max_epochs': args_train.max_epochs,
            'batch_size': args_train.batch_size,
            'virtual_batch_size': args_train.virtual_batch_size,
            'learning_rate': model.optimizer_params['lr'],
            'scheduler_step_size': model.scheduler_params['step_size']
        })


        model.fit(**dict_train)

        # 测试集预测，注意：predict_proba函数只能得到每个样本在每个类上的softmax概率，可以直接算auc，算准确率要转换一下
        loaded_preds = model.predict_proba(X_test)
        # 评估模型
        # 算auc
        auc = roc_auc_score(y_score=loaded_preds[:, 1], y_true=y_test)
        # 算acc
        acc = accuracy_score(y_pred=loaded_preds.argmax(1), y_true=y_test)
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
        # print("preds_A", loaded_preds[:,1])
        print(f"acc: {acc:.2%}, "
              f"auc: {auc:.2%}, "
              f"sensitivity: {sensitivity:.2%}, "
              f"specificity: {specificity:.2%}, "
              f"f-measure: {fs:.2%}, "
              f"MCC: {mc:.2%}")
        # 记录结果
        results.append({
            'iteration': i + 1,
            'auc': auc,
            'acc': acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f-measure': fs,
            'mcc': mc
        })

        # 将最优参数列表转换为DataFrame
        optimized_params_df = pd.DataFrame(optimized_params_list)
        # 保存最优参数到CSV文件
        optimized_params_df.to_csv('optimized_params.csv', index=False)
        print("Optimized Parameters saved to 'optimized_params.csv'")
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(results)
        # 保存到CSV文件
        results_df.to_csv('pso_results.csv', index=False)
        print("Results saved to 'pso_results.csv'")

    # 计算评估指标的平均值
    average_metrics = results_df.mean()
    print("Average Metrics:")
    print(f"AUC: {average_metrics['auc']:.4f}")
    print(f"Accuracy: {average_metrics['acc']:.4f}")
    print(f"Sensitivity: {average_metrics['sensitivity']:.4f}")
    print(f"Specificity: {average_metrics['specificity']:.4f}")
    print(f"F-measure: {average_metrics['f-measure']:.4f}")
    print(f"MCC: {average_metrics['mcc']:.4f}")

