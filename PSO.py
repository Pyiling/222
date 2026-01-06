import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def fitness_function(params, model, X_train, y_train, X_valid, y_valid, train_arg_para):
    """
    评估函数，根据给定的超参数评估模型在验证集上的AUC。

    参数:
    - params: 要优化的超参数列表
    - model: TabNetClassifier 实例
    - X_train, y_train: 训练数据
    - X_valid, y_valid: 验证数据

    返回:
    - auc: 模型在验证集上的AUC分数
    """
    # 解包参数
    (n_d, n_a, lambda_sparse, gamma, n_steps, n_shared,
     max_epochs, batch_size, virtual_batch_size, learning_rate, step_size) = params
    # 更新模型的超参数
    model.set_params(
        n_d=int(n_d),
        n_a=int(n_a),
        lambda_sparse=lambda_sparse,
        gamma=gamma,
        n_steps=int(n_steps),
        n_shared=int(n_shared),
        optimizer_params={'lr': learning_rate},
        scheduler_params={'step_size': step_size},
    )
    dict_train = train_arg_para
    dict_train['max_epochs'] = int(max_epochs)
    dict_train['batch_size'] = int(batch_size)
    dict_train['virtual_batch_size'] = int(virtual_batch_size)
    # 训练模型
    model.fit(**dict_train)
    # 预测验证集
    preds_auc = model.predict_proba(X_valid)[:, 1]
    preds_acc = model.predict_proba(X_valid).argmax(1)

    # 计算AUC
    acc = accuracy_score(y_valid, preds_acc)
    auc = roc_auc_score(y_valid, preds_auc)

    return acc

def pso_optimize(fitness_func, model, X_train, y_train, X_valid, y_valid,train_arg_para, n_particles=20, max_iters=50, w=0.5, c1=1.5, c2=1.5):
    """
    粒子群优化算法，用于优化模型的超参数。
    参数:
    - fitness_func: 适应度函数
    - model: TabNetClassifier 实例
    - X_train, y_train: 训练数据
    - X_valid, y_valid: 验证数据
    - n_particles: 粒子数量
    - max_iters: 最大迭代次数
    - w: 惯性权重
    - c1: 认知系数
    - c2: 社会系数
    返回:
    - g_best: 最佳粒子的参数
    - g_best_score: 最佳粒子的适应度值
    """
    # 定义搜索空间的边界
    bounds = {
        'n_d': (4, 64),
        'n_a': (4, 64),
        'lambda_sparse': (1e-6, 1e-3),
        'gamma': (1.0, 1.4),
        'n_steps': [1,2,3,4,5,6],
        'n_shared': [1,2,3],
        'max_epochs': [128,256,512],
        'batch_size': [32,64,128],
        'virtual_batch_size': [16,32,64],
        'learning_rate': (1e-4, 1e-1),
        'step_size': (1,50),
    }

    param_names = list(bounds.keys())
    n_params = len(param_names)

    # 初始化粒子的位置和速度
    particles = np.zeros((n_particles, n_params))
    for i, param in enumerate(param_names):
        if isinstance(bounds[param], list):
            # 如果是离散的，随机从集合中选取
            particles[:, i] = np.random.choice(bounds[param], size=n_particles)
        else:
            # 如果是连续的，使用uniform进行随机初始化
            particles[:, i] = np.random.uniform(bounds[param][0], bounds[param][1], size=n_particles)

    velocity = np.random.uniform(low=-1, high=1, size=(n_particles, n_params))

    # 初始化个人最优解和全局最优解
    p_best = particles.copy()
    p_best_scores = np.array([fitness_func(p, model, X_train, y_train, X_valid, y_valid, train_arg_para) for p in particles])
    g_best_idx = np.argmax(p_best_scores)
    g_best = particles[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]

    for iter in range(max_iters):
        for i in range(n_particles):
            # 计算适应度
            fitness = fitness_func(particles[i], model, X_train, y_train, X_valid, y_valid, train_arg_para)

            # 更新个人最优
            if fitness > p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = fitness

                # 更新全局最优
                if fitness > g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = fitness

        # 更新粒子的速度和位置
        r1 = np.random.rand(n_particles, n_params)
        r2 = np.random.rand(n_particles, n_params)

        velocity = (w * velocity +
                    c1 * r1 * (p_best - particles) +
                    c2 * r2 * (g_best - particles))

        particles += velocity

        # 确保粒子的位置在边界内
        for i, param in enumerate(param_names):
            if isinstance(bounds[param], list):
                # 离散变量处理，强制保持在离散集合中
                particles[:, i] = np.random.choice(bounds[param], size=1)
            else:
                # 连续变量处理，确保其在边界内
                particles[:, i] = np.clip(particles[:, i], bounds[param][0], bounds[param][1])

        print(f"Iteration {iter+1}/{max_iters}, Best AUC: {g_best_score:.4f}")

    return g_best, g_best_score
