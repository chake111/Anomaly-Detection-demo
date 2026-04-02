# Anomaly Detection Demo

本仓库当前用于异常检测练习代码演示。

## 1) 创建环境

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) 运行现有脚本

```bash
python main.py
```

## 3) 运行 Notebook 练习（适配题目中的环境）

```bash
jupyter notebook
```

启动后可在浏览器中打开并创建/运行你的练习 Notebook（含 `numpy`、`matplotlib`、`sklearn`、`ipykernel` 环境）。

建议完成第一次练习并保存为：`practice_01.ipynb`。

## 4) 保存与复用

- 首次练习完成后保存 Notebook：`practice_01.ipynb`
- 后续每次练习前，先激活虚拟环境再启动 Notebook：

```bash
source .venv/bin/activate
jupyter notebook
```

## 5) 自检通过标准

- 能成功启动 Notebook
- 能导入 `numpy`、`matplotlib`、`sklearn`
- 能完成一次“生成数据 → 训练（IsolationForest） → 预测 → 可视化”
