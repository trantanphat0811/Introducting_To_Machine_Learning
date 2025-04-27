from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(11)

# Tạo dữ liệu các điểm theo 3 cụm
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]  # Ma trận đơn vị
N = 500  # Số điểm dữ liệu mỗi cụm

# Tạo dữ liệu phân phối chuẩn đa biến
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# Ghép dữ liệu thành một tập duy nhất
X = np.concatenate((X0, X1, X2), axis=0)

# Số cụm
K = 3  
# Gán nhãn ban đầu cho từng cụm
original_label = np.repeat([0, 1, 2], N)

# Hàm hiển thị dữ liệu
def kmeans_display(X, label):
    K = np.amax(label) + 1  # Xác định số cụm
    colors = ['b', 'g', 'r']
    markers = ['^', 'o', 's']
    
    plt.figure(figsize=(8, 6))
    for k in range(K):
        cluster_points = X[np.where(label == k)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    c=colors[k], marker=markers[k], label=f'Cluster {k}', alpha=0.6)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Visualization of Initial Data Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

# Hiển thị dữ liệu
kmeans_display(X, original_label)

# Đoạn code tạo ra 3 cụm dữ liệu ngẫu nhiên phân phối chuẩn đa biến
# Mỗi cụm có trung tâm khác nhau:
# Cụm 1: (2, 2)
# Cụm 2: (8, 3)
# Cụm 3: (3, 6)
# Sử dụng ma trận hiệp phương sai (cov) đơn vị