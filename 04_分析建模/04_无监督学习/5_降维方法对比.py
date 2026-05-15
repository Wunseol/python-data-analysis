# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集), sklearn.datasets.load_digits (手写数字数据集)

"""
降维方法对比
============
本案例演示:
1. PCA vs t-SNE vs UMAP 概念对比
2. PCA 重建 (逆变换)
3. IncrementalPCA 增量主成分分析
4. KernelPCA 核主成分分析 (linear/rbf/poly 核)
5. t-SNE 可视化 (perplexity 参数)
6. 多种降维方法在 iris/digits 数据集上的对比
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、数据加载与预处理
# ============================================================

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)

print("=" * 60)
print("【数据集信息】")
print(f"鸢尾花: 样本数={X_iris.shape[0]}, 特征数={X_iris.shape[1]}, 类别数={len(np.unique(y_iris))}")
print(f"手写数字: 样本数={X_digits.shape[0]}, 特征数={X_digits.shape[1]}, 类别数={len(np.unique(y_digits))}")

# ============================================================
# 二、PCA 重建 (逆变换)
# ============================================================

pca_full = PCA()
X_iris_pca_full = pca_full.fit_transform(X_iris_scaled)

n_components_list = [1, 2, 3, 4]
print("\n" + "=" * 60)
print("【PCA 重建质量 (鸢尾花数据集)】")
print(f"{'主成分数':<10} {'累积方差解释率':>18} {'重建MSE':>12}")
print("-" * 42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, n_c in enumerate(n_components_list):
    pca_rc = PCA(n_components=n_c)
    X_reduced = pca_rc.fit_transform(X_iris_scaled)
    X_reconstructed = pca_rc.inverse_transform(X_reduced)

    mse = np.mean((X_iris_scaled - X_reconstructed) ** 2)
    cum_var = pca_rc.explained_variance_ratio_.sum()

    print(f"{n_c:<10} {cum_var:>18.4f} {mse:>12.6f}")

    ax = axes[idx // 2, idx % 2]
    ax.scatter(X_iris_scaled[:, 0], X_iris_scaled[:, 1],
               c='steelblue', s=30, alpha=0.5, label='原始数据')
    ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
               c='darkorange', s=30, alpha=0.5, marker='s', label='重建数据')
    ax.set_title(f'n_components={n_c}, 累积方差={cum_var:.2%}, MSE={mse:.4f}', fontsize=11)
    ax.set_xlabel('特征1 (标准化)')
    ax.set_ylabel('特征2 (标准化)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('PCA 重建: 不同主成分数量的重建效果', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_PCA重建.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 三、IncrementalPCA 增量主成分分析
# ============================================================

n_batches = 10
batch_size = len(X_digits_scaled) // n_batches

ipca = IncrementalPCA(n_components=2, batch_size=batch_size)

for batch_idx in range(n_batches):
    start = batch_idx * batch_size
    end = start + batch_size
    ipca.partial_fit(X_digits_scaled[start:end])

X_digits_ipca = ipca.transform(X_digits_scaled)

pca_2d_digits = PCA(n_components=2)
X_digits_pca_2d = pca_2d_digits.fit_transform(X_digits_scaled)

print("\n" + "=" * 60)
print("【IncrementalPCA vs PCA (手写数字数据集)】")
print(f"IncrementalPCA 方差解释率: {ipca.explained_variance_ratio_}")
print(f"PCA 方差解释率:           {pca_2d_digits.explained_variance_ratio_}")
print(f"IncrementalPCA 累积:      {ipca.explained_variance_ratio_.sum():.4f}")
print(f"PCA 累积:                 {pca_2d_digits.explained_variance_ratio_.sum():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(X_digits_pca_2d[:, 0], X_digits_pca_2d[:, 1],
                            c=y_digits, cmap=plt.cm.tab10, s=10, alpha=0.6)
axes[0].set_title('PCA 2D (手写数字)', fontsize=13)
axes[0].set_xlabel(f'PC1 ({pca_2d_digits.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca_2d_digits.explained_variance_ratio_[1]:.2%})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='数字类别')

scatter2 = axes[1].scatter(X_digits_ipca[:, 0], X_digits_ipca[:, 1],
                            c=y_digits, cmap=plt.cm.tab10, s=10, alpha=0.6)
axes[1].set_title('IncrementalPCA 2D (手写数字)', fontsize=13)
axes[1].set_xlabel(f'PC1 ({ipca.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'PC2 ({ipca.explained_variance_ratio_[1]:.2%})')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='数字类别')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_IncrementalPCA.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 四、KernelPCA 核主成分分析
# ============================================================

kernels = ['linear', 'rbf', 'poly']
kpca_results = {}

print("\n" + "=" * 60)
print("【KernelPCA 不同核函数 (鸢尾花数据集)】")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, kernel in enumerate(kernels):
    if kernel == 'rbf':
        kpca = KernelPCA(n_components=2, kernel=kernel, gamma=0.5)
    elif kernel == 'poly':
        kpca = KernelPCA(n_components=2, kernel=kernel, degree=3, gamma=1)
    else:
        kpca = KernelPCA(n_components=2, kernel=kernel)

    X_kpca = kpca.fit_transform(X_iris_scaled)
    kpca_results[kernel] = X_kpca

    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y_iris == cls_idx
        axes[idx].scatter(X_kpca[mask, 0], X_kpca[mask, 1],
                          label=cls_name, s=40, alpha=0.7)
    axes[idx].set_title(f'KernelPCA (kernel={kernel})', fontsize=13)
    axes[idx].set_xlabel('KPC1')
    axes[idx].set_ylabel('KPC2')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

    print(f"  kernel={kernel}: 形状={X_kpca.shape}")

plt.suptitle('KernelPCA: 不同核函数对比', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_KernelPCA.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 五、KernelPCA gamma 参数影响
# ============================================================

gamma_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

print("\n" + "=" * 60)
print("【KernelPCA RBF核: gamma 参数影响 (鸢尾花数据集)】")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, gamma in enumerate(gamma_values):
    kpca_g = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    X_kpca_g = kpca_g.fit_transform(X_iris_scaled)

    ax = axes[idx // 3, idx % 3]
    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y_iris == cls_idx
        ax.scatter(X_kpca_g[mask, 0], X_kpca_g[mask, 1],
                   label=cls_name, s=30, alpha=0.7)
    ax.set_title(f'gamma={gamma}', fontsize=13)
    ax.set_xlabel('KPC1')
    ax.set_ylabel('KPC2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('KernelPCA (RBF核): gamma 参数对降维结果的影响', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_KernelPCA_gamma.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 六、t-SNE 可视化
# ============================================================

perplexity_values = [5, 10, 20, 30, 40, 50]

print("\n" + "=" * 60)
print("【t-SNE: perplexity 参数影响 (鸢尾花数据集)】")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, perp in enumerate(perplexity_values):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000)
    X_tsne = tsne.fit_transform(X_iris_scaled)

    ax = axes[idx // 3, idx % 3]
    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y_iris == cls_idx
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=cls_name, s=40, alpha=0.7)
    ax.set_title(f'perplexity={perp}', fontsize=13)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('t-SNE: perplexity 参数对降维结果的影响', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_tSNE_perplexity.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、手写数字数据集 t-SNE 可视化
# ============================================================

print("\n" + "=" * 60)
print("【t-SNE 可视化手写数字数据集 (perplexity=30)】")

tsne_digits = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_digits_tsne = tsne_digits.fit_transform(X_digits_scaled)

fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1],
                      c=y_digits, cmap=plt.cm.tab10, s=15, alpha=0.7)
plt.colorbar(scatter, ax=ax, label='数字类别')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE 可视化: 手写数字数据集', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_tSNE_digits.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 八、多种降维方法综合对比 (鸢尾花数据集)
# ============================================================

pca_iris_2d = PCA(n_components=2).fit_transform(X_iris_scaled)
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.5).fit_transform(X_iris_scaled)
tsne_iris = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(X_iris_scaled)

methods_iris = {
    'PCA': pca_iris_2d,
    'KernelPCA (rbf)': kpca_rbf,
    't-SNE': tsne_iris,
}

print("\n" + "=" * 60)
print("【多种降维方法对比 (鸢尾花数据集)】")
print(f"{'方法':<20} {'轮廓系数':>10}")
print("-" * 32)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, X_2d) in enumerate(methods_iris.items()):
    sil = silhouette_score(X_2d, y_iris)
    print(f"{name:<20} {sil:>10.4f}")

    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y_iris == cls_idx
        axes[idx].scatter(X_2d[mask, 0], X_2d[mask, 1],
                          label=cls_name, s=40, alpha=0.7)
    axes[idx].set_title(f'{name} (轮廓系数={sil:.4f})', fontsize=13)
    axes[idx].set_xlabel('维度1')
    axes[idx].set_ylabel('维度2')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('鸢尾花数据集: 多种降维方法对比', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_鸢尾花综合.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 九、多种降维方法综合对比 (手写数字数据集)
# ============================================================

pca_digits_2d = PCA(n_components=2).fit_transform(X_digits_scaled)
kpca_digits = KernelPCA(n_components=2, kernel='rbf', gamma=0.01).fit_transform(X_digits_scaled)

methods_digits = {
    'PCA': pca_digits_2d,
    'KernelPCA (rbf)': kpca_digits,
    't-SNE': X_digits_tsne,
}

print("\n" + "=" * 60)
print("【多种降维方法对比 (手写数字数据集)】")
print(f"{'方法':<20} {'轮廓系数':>10}")
print("-" * 32)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (name, X_2d) in enumerate(methods_digits.items()):
    sil = silhouette_score(X_2d, y_digits)
    print(f"{name:<20} {sil:>10.4f}")

    scatter = axes[idx].scatter(X_2d[:, 0], X_2d[:, 1],
                                 c=y_digits, cmap=plt.cm.tab10, s=10, alpha=0.6)
    axes[idx].set_title(f'{name} (轮廓系数={sil:.4f})', fontsize=13)
    axes[idx].set_xlabel('维度1')
    axes[idx].set_ylabel('维度2')
    axes[idx].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[idx], label='数字类别')

plt.suptitle('手写数字数据集: 多种降维方法对比', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '降维对比_digits综合.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 十、UMAP (可选, 需要安装 umap-learn)
# ============================================================

try:
    from umap import UMAP

    umap_iris = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_iris_umap = umap_iris.fit_transform(X_iris_scaled)

    umap_digits = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_digits_umap = umap_digits.fit_transform(X_digits_scaled)

    print("\n" + "=" * 60)
    print("【UMAP 降维结果】")
    print(f"鸢尾花 UMAP 轮廓系数: {silhouette_score(X_iris_umap, y_iris):.4f}")
    print(f"手写数字 UMAP 轮廓系数: {silhouette_score(X_digits_umap, y_digits):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y_iris == cls_idx
        axes[0].scatter(X_iris_umap[mask, 0], X_iris_umap[mask, 1],
                        label=cls_name, s=40, alpha=0.7)
    axes[0].set_title('UMAP - 鸢尾花数据集', fontsize=13)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(X_digits_umap[:, 0], X_digits_umap[:, 1],
                               c=y_digits, cmap=plt.cm.tab10, s=10, alpha=0.6)
    axes[1].set_title('UMAP - 手写数字数据集', fontsize=13)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='数字类别')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '降维对比_UMAP.png', dpi=150, bbox_inches='tight')
    plt.show()

except ImportError:
    print("\n" + "=" * 60)
    print("【UMAP 未安装】")
    print("UMAP 库未安装, 如需使用请执行: pip install umap-learn")
    print("UMAP 特点: 保留全局和局部结构, 速度比 t-SNE 快, 适合大规模数据")

# ============================================================
# 十一、降维方法特性总结
# ============================================================

print("\n" + "=" * 60)
print("【降维方法特性总结】")
print(f"{'方法':<20} {'线性/非线性':>12} {'监督/无监督':>12} {'适用场景':>20}")
print("-" * 68)
print(f"{'PCA':<20} {'线性':>12} {'无监督':>12} {'特征压缩/去噪':>20}")
print(f"{'IncrementalPCA':<20} {'线性':>12} {'无监督':>12} {'大规模数据':>20}")
print(f"{'KernelPCA':<20} {'非线性':>12} {'无监督':>12} {'非线性结构':>20}")
print(f"{'t-SNE':<20} {'非线性':>12} {'无监督':>12} {'可视化(2D/3D)':>20}")
print(f"{'UMAP':<20} {'非线性':>12} {'无监督':>12} {'可视化/特征提取':>20}")

print("\n" + "=" * 60)
print("降维方法对比演示完成!")
