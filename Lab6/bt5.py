import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs

def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Gán nhãn cho mỗi điểm dựa trên khoảng cách gần nhất
        labels = pairwise_distances_argmin(X, centers)

        # Tính toán tâm cụm mới
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] 
                                for i in range(n_clusters)])

        # Dừng nếu không có sự thay đổi về tâm cụm
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

# Sinh dữ liệu ngẫu nhiên với 4 cụm
X, y = make_blobs(n_samples=800, n_features=2, centers=4, random_state=23)

# Áp dụng thuật toán K-Means
centers, labels = find_clusters(X, 4)

# Hiển thị kết quả phân cụm
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', label="Điểm dữ liệu")

# Hiển thị tâm cụm
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label="Tâm cụm")

# Thêm thông tin biểu đồ
plt.title("Phân cụm K-Means từ đầu", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

#Biểu đồ các cụm dữ liệu được tô màu và tâm cụm được đánh dấu bằng dấu X màu đỏ
# Sử dụng hàm pairwise_distances_argmin để tìm khoảng cách gần nhất giữa các điểm dữ liệu và tâm cụm
# Sử dụng hàm np.random.RandomState để tạo ra các số ngẫu nhiên