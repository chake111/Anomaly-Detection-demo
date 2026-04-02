import numpy as np
import matplotlib.pyplot as plt


def load_data(seed=2):
    """
    生成二维异常检测数据
    返回:
        X_train: (307, 2)
        X_val:   (307, 2)
        y_val:   (307,)
    """
    rng = np.random.default_rng(seed)

    # 正常训练数据
    mu_train = np.array([14.0, 15.0])
    std_train = np.array([1.35, 1.30])
    X_train = rng.normal(loc=mu_train, scale=std_train, size=(307, 2))

    # 验证集：大部分正常，小部分异常
    n_val = 307
    n_anom = 20
    n_norm = n_val - n_anom

    X_val_normal = rng.normal(loc=mu_train, scale=std_train, size=(n_norm, 2))

    # 生成离群异常点
    anomaly_centers = np.array([
        [3, 3],
        [25, 25],
        [3, 25],
        [25, 3],
        [8, 24],
        [24, 8]
    ])
    chosen = anomaly_centers[rng.integers(0, len(anomaly_centers), size=n_anom)]
    X_val_anom = chosen + rng.normal(0, 1.2, size=(n_anom, 2))

    X_val = np.vstack([X_val_normal, X_val_anom])
    y_val = np.hstack([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])

    # 打乱验证集
    perm = rng.permutation(n_val)
    X_val = X_val[perm]
    y_val = y_val[perm]

    return X_train, X_val, y_val


def load_data_multi(seed=11):
    """
    生成高维异常检测数据
    返回:
        X_train_high: (1000, 11)
        X_val_high:   (100, 11)
        y_val_high:   (100,)
    """
    rng = np.random.default_rng(seed)

    n_features = 11

    # 为每个特征设置不同均值和标准差
    mu = np.array([10, 12, 14, 9, 11, 13, 15, 10.5, 12.5, 14.5, 16])
    std = np.array([1.2, 1.5, 1.1, 1.4, 1.3, 1.6, 1.0, 1.7, 1.2, 1.4, 1.1])

    X_train_high = rng.normal(loc=mu, scale=std, size=(1000, n_features))

    n_val = 100
    n_anom = 15
    n_norm = n_val - n_anom

    X_val_normal = rng.normal(loc=mu, scale=std, size=(n_norm, n_features))

    # 高维异常：偏移多个维度
    anomaly_shift = rng.choice([-8, -6, 6, 8], size=(n_anom, n_features))
    noise = rng.normal(0, 1.5, size=(n_anom, n_features))
    X_val_anom = mu + anomaly_shift + noise

    X_val_high = np.vstack([X_val_normal, X_val_anom])
    y_val_high = np.hstack([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])

    perm = rng.permutation(n_val)
    X_val_high = X_val_high[perm]
    y_val_high = y_val_high[perm]

    return X_train_high, X_val_high, y_val_high


def multivariate_gaussian(X, mu, var):
    """
    计算样本在多元高斯分布下的概率密度

    参数:
        X:   (m, n)
        mu:  (n,)
        var: (n,) 或 (n, n)

    返回:
        p:   (m,)
    """
    X = np.asarray(X)
    mu = np.asarray(mu)
    var = np.asarray(var)

    n = len(mu)

    if var.ndim == 1:
        Sigma = np.diag(var)
    else:
        Sigma = var

    # 数值稳定性处理
    Sigma = Sigma + 1e-6 * np.eye(n)

    X_mu = X - mu
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)

    coeff = 1.0 / (((2 * np.pi) ** (n / 2)) * np.sqrt(det))
    exp_term = np.sum((X_mu @ inv) * X_mu, axis=1)

    p = coeff * np.exp(-0.5 * exp_term)
    return p

def visualize_fit(X, mu, var):
    plt.figure(figsize=(6, 6))
    plt.plot(X[:, 0], X[:, 1], 'bx')

    x = np.arange(0, 35.5, 0.5)
    y = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(x, y)

    Z = np.zeros(X1.shape)
    points = np.column_stack((X1.ravel(), X2.ravel()))
    Z = multivariate_gaussian(points, mu, var).reshape(X1.shape)

    plt.contour(X1, X2, Z, levels=[10**h for h in range(-20, 0, 3)], linewidths=1)

    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.title('The Gaussian contours of the distribution fit to the dataset')
    plt.axis([0, 35, 0, 35])
