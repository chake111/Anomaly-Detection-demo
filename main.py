import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

# 模拟数据
# 正常数据
norm_data = np.random.randn(300, 2)

# 异常数据
anomalies = np.array([[2.5, 2.8], [3.0, 2.5], [-2.8, -2.5], [2.7, -2.6], [-2.9, 2.6]])

X = np.vstack([norm_data, anomalies])

plt.figure(figsize=(6, 6))
plt.scatter(norm_data[:, 0], norm_data[:, 1], label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], label='Anomaly', color='red', s=80)
plt.legend()
plt.show()

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

z_scores = np.abs((X - mean) / std)

z_pred = (z_scores > 3).any(axis=1)

print("Z-Score 检测出的异常点索引:\n", np.where(z_pred)[0])

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=z_pred, cmap='coolwarm', s=40)
plt.title("Z-Score Anomaly Detection")
plt.show()

lof = LocalOutlierFactor(n_neighbors=50, contamination=0.02)
lof_pred = lof.fit_predict(X)

lof_pred = (lof_pred == -1)
print("LOF 检测出的异常点索引:\n", np.where(lof_pred)[0])

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=lof_pred, cmap='coolwarm', s=40)
plt.title("LOF Anomaly Detection")
plt.show()

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
iso.fit(X)
iso_pred = iso.predict(X)

# 正常是 1，异常是 -1
iso_pred = (iso_pred == -1)

print("Isolation Forest 检测出的异常点索引：")
print(np.where(iso_pred)[0])

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=iso_pred, cmap='coolwarm', s=40)
plt.title("Isolation Forest")
plt.show()

print("Z-Score 异常数量:", z_pred.sum())
print("LOF 异常数量:", lof_pred.sum())
print("Isolation Forest 异常数量:", iso_pred.sum())

y_true = np.array([0] * len(norm_data) + [1] * len(anomalies))

from sklearn.metrics import classification_report

print("Z-Score")
print(classification_report(y_true, z_pred.astype(int)))

print("LOF")
print(classification_report(y_true, lof_pred.astype(int)))

print("Isolation Forest")
print(classification_report(y_true, iso_pred.astype(int)))
