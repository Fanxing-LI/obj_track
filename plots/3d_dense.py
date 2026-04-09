import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.datasets import make_blobs  # 方便生成混合高斯分布

# 1. 生成三维点云（混合高斯分布）
n_samples = 2000
centers = [[-2, -2, -2], [2, 2, 2]]   # 两个簇的中心
cluster_std = [1.2, 0.8]               # 两个簇的标准差
X, _ = make_blobs(n_samples=n_samples, centers=centers, 
                  cluster_std=cluster_std, n_features=3, random_state=42)

# 2. 核密度估计
kde = gaussian_kde(X.T)  # 注意：kde 需要数据形状为 (n_features, n_samples)

# 3. 创建三维网格
# 获取点云范围，并向外扩展一点
x_min, y_min, z_min = X.min(axis=0) - 1
x_max, y_max, z_max = X.max(axis=0) + 1

# 网格分辨率（各维度点数，可根据需要调整）
grid_res = 30
x_grid = np.linspace(x_min, x_max, grid_res)
y_grid = np.linspace(y_min, y_max, grid_res)
z_grid = np.linspace(z_min, z_max, grid_res)
X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

# 将网格点展平为坐标列表
grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

# 计算每个网格点的密度值
density = kde(grid_points).reshape(X_grid.shape)

# 4. 可视化基础数据处理
point_densities = kde(X.T)

# 5. 生成 2D KDE 用于投影
# XZ 平面
kde_xz = gaussian_kde(X[:, [0, 2]].T)
xz_grid = np.vstack([X_grid[:,0,:].ravel(), Z_grid[:,0,:].ravel()])
density_xz = kde_xz(xz_grid).reshape(X_grid[:,0,:].shape)
point_densities_xz = kde_xz(X[:, [0, 2]].T)

# XY 平面
kde_xy = gaussian_kde(X[:, [0, 1]].T)
xy_grid = np.vstack([X_grid[:,:,0].ravel(), Y_grid[:,:,0].ravel()])
density_xy = kde_xy(xy_grid).reshape(X_grid[:,:,0].shape)
point_densities_xy = kde_xy(X[:, [0, 1]].T)

# 6. 使用 Matplotlib 创建子图 (3行2列)
fig = plt.figure(figsize=(10, 15))

# 第1行, 第1列: 3D 原始点云
ax1 = fig.add_subplot(3, 2, 1, projection='3d')
sc1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_densities, cmap='viridis', s=10)
ax1.set_title('3D 原始点云')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 第1行, 第2列: 3D 密度分布 (使用半透明散点模拟)
ax2 = fig.add_subplot(3, 2, 2, projection='3d')
density_max = density.max()
mask = density > (0.2 * density_max)
sc2 = ax2.scatter(X_grid[mask], Y_grid[mask], Z_grid[mask], 
                  c=density[mask], cmap='Reds', alpha=0.3, s=15)
ax2.set_title('3D 点云密度 (Density > 20% max)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# 第2行, 第1列: XZ 平面点云
ax3 = fig.add_subplot(3, 2, 3)
sc3 = ax3.scatter(X[:, 0], X[:, 2], c=point_densities_xz, cmap='viridis', s=10)
ax3.set_title('XZ 平面点云')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')

# 第2行, 第2列: XZ 平面密度 (等高线)
ax4 = fig.add_subplot(3, 2, 4)
ct4 = ax4.contourf(x_grid, z_grid, density_xz.T, levels=20, cmap='Reds')
ax4.set_title('XZ 平面密度')
ax4.set_xlabel('X')
ax4.set_ylabel('Z')

# 第3行, 第1列: XY 平面点云
ax5 = fig.add_subplot(3, 2, 5)
sc5 = ax5.scatter(X[:, 0], X[:, 1], c=point_densities_xy, cmap='viridis', s=10)
ax5.set_title('XY 平面点云')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')

# 第3行, 第2列: XY 平面密度 (等高线)
ax6 = fig.add_subplot(3, 2, 6)
ct6 = ax6.contourf(x_grid, y_grid, density_xy.T, levels=20, cmap='Reds')
ax6.set_title('XY 平面密度')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')

plt.tight_layout()
plt.savefig("3d_density_plot.png", dpi=300)
print("Saved map rendering to 3d_density_plot.png")
