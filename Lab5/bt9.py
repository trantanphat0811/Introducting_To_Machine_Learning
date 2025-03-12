import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D  # Không bắt buộc, nhưng có thể giúp rõ ràng hơn

# Tạo dữ liệu vòng tròn
X, y = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=0)

# Biến đổi dữ liệu thành không gian 3D
X1 = X[:, 0].reshape(-1, 1)  # Cột 1
X2 = X[:, 1].reshape(-1, 1)  # Cột 2
X3 = (X1**2 + X2**2)  # Biến mới X3 = X1^2 + X2^2
X_transformed = np.hstack((X, X3))  # Kết hợp dữ liệu mới

# Vẽ dữ liệu trong không gian 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ scatter plot với trục 3D
ax.scatter(X1, X2, X3, c=y, cmap='spring', depthshade=True)

# Thiết lập nhãn trục
ax.set_xlabel("Feature 1 (X1)")
ax.set_ylabel("Feature 2 (X2)")
ax.set_zlabel("X1^2 + X2^2 (X3)")
ax.set_title("Phân bố dữ liệu sau khi ánh xạ lên không gian 3D")

plt.show()
#Biểu đồ 3D hiển thị dữ liệu vòng tròn sau khi biến đổi lên không gian 3D.
#Dữ liệu ban đầu có 2 đặc trưng (X1, X2) được biến đổi thành 3 đặc trưng (X1, X2, X3 = X1^2 + X2^2).
#Dữ liệu được tô màu theo nhãn y để phân biệt giữa 2 lớp dữ liệu.