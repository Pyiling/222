import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import AdaBoostClassifier as ADA, GradientBoostingClassifier as GBDT, RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.gaussian_process import GaussianProcessClassifier as GPC

# 数据路径
data_train_path = r"D:\python\git\TableNet-main\trainnew_mri_data.csv"
data_test_path = r"D:\python\git\TableNet-main\testnew_mri_data.csv"

# 读取数据
X_train = pd.read_csv(data_train_path).values[:, :-1]
y_train = pd.read_csv(data_train_path).values[:, -1]
X_test = pd.read_csv(data_test_path).values[:, :-1]
y_test = pd.read_csv(data_test_path).values[:, -1]

# 定义模型
models = {
    'LR': LR(),
    'ADA': ADA(),
    'DT': DT(),
    'GBDT': GBDT(),
    'GNB': GNB(),
    'GPC': GPC(),
    'KNN': KNN(),
    'LDA': LDA(),
    'MLP': MLP(),
    'RF': RF(),
    'SVM': SVM(probability=True)
}


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}

    for name, model in models.items():
        # 拟合模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 计算指标
        acc = accuracy_score(y_test, y_pred) * 100
        auc_score = roc_auc_score(y_test, y_prob) * 100
        f1 = f1_score(y_test, y_pred) * 100
        mcc = matthews_corrcoef(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) * 100
        specificity = tn / (tn + fp) * 100

        results[name] = {
            'ACC (%)': f"{acc:.2f}%",
            'AUC (%)': f"{auc_score:.2f}%",
            'Sensitivity (%)': f"{sensitivity:.2f}%",
            'Specificity (%)': f"{specificity:.2f}%",
            'F1-Score (%)': f"{f1:.2f}%",
            'MCC': f"{mcc:.2f}"  # MCC 通常不以百分数表示
        }

    return results


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 10))  # 设置为正方形图形

    for name, model in models.items():
        # 预测概率
        y_prob = model.predict_proba(X_test)[:, 1]

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # 绘制 ROC 曲线
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)

    # 图形设置
    plt.title('临床模型AUC曲线', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=8, frameon=False)  # 进一步缩小图例字体
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # 设置正方形的框
    plt.gca().set_aspect('equal', adjustable='box')

    # 保持刻度
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置x轴刻度
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置y轴刻度

    plt.grid(False)  # 显示网格以增强可读性
    plt.show()


# 评估模型并输出结果
results = evaluate_models(models, X_train, y_train, X_test, y_test)
results_df = pd.DataFrame(results).T
print(results_df)

# 绘制 ROC 曲线
plot_roc_curves(models, X_test, y_test)

TABNETAUC临床
fpr:[0.         0.         0.         0.         0.         0.
 0.0212766  0.0212766  0.0212766  0.0212766  0.0212766  0.0212766
 0.04255319 0.04255319 0.04255319 0.04255319 0.04255319 0.06382979
 0.06382979 0.06382979 0.06382979 0.06382979 0.06382979 0.06382979
 0.06382979 0.06382979 0.06382979 0.10638298 0.12765957 0.12765957
 0.12765957 0.12765957 0.14893617 0.14893617 0.17021277 0.17021277
 0.19148936 0.19148936 0.21276596 0.21276596 0.23404255 0.44680851
 0.44680851 0.46808511 0.46808511 0.55319149 0.59574468 1.        ]
tpr:[0.         0.06338028 0.07042254 0.14084507 0.19014085 0.20422535
 0.20422535 0.23239437 0.27464789 0.28169014 0.29577465 0.30985915
 0.36619718 0.38732394 0.41549296 0.43661972 0.45070423 0.45070423
 0.45774648 0.49295775 0.51408451 0.52816901 0.54225352 0.55633803
 0.57746479 0.59859155 0.65492958 0.65492958 0.66901408 0.6971831
 0.71126761 0.76056338 0.76056338 0.76760563 0.76760563 0.77464789
 0.78873239 0.79577465 0.82394366 0.83098592 0.87323944 0.91549296
 0.92253521 0.92253521 0.93661972 0.94366197 0.94366197 1.        ]
roc_auc:0.8705424033563081
FEATURE IMPORTANCES FOR axpan : [0.07464016 0.12427455 0.04035648 0.08696772 0.12918267 0.16471094
 0.08084174 0.04436423 0.05536443 0.07746562 0.07039181 0.05143965]

