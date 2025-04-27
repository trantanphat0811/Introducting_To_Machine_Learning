import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Tạo dữ liệu giả lập với 4 cụm
X, y = make_blobs(n_samples=800, n_features=2, centers=4, random_state=23)

# Khởi tạo và huấn luyện mô hình KMeans
kmeans = KMeans(n_clusters=4, n_init=10, random_state=23)  # Fix warning với n_init=10
kmeans.fit(X)

# Dự đoán cụm của từng điểm dữ liệu
y_kmeans = kmeans.predict(X)

# Vẽ scatter plot của dữ liệu, tô màu theo cụm
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis', label="Dữ liệu phân cụm")

# Vẽ tâm cụm
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label="Tâm cụm")

# Thêm tiêu đề và nhãn trục
plt.title("K-Means Clustering với 4 cụm", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Dữ liệu được tạo ra với 4 cụm khác nhau ,mỗi cụm có màu sắc khác nhau
# Tâm cụm được đánh dấu bằng dấu X màu đỏ
# Các điểm nằm gần nhau được phân nhóm thành cùng một cụm