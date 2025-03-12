import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Tạo dữ liệu mẫu với 2 cụm (clusters)
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=0, cluster_std=0.40)

# Vẽ biểu đồ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring', edgecolors='k')

# Thêm tiêu đề và nhãn trục
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot của Dữ liệu Tạo bằng make_blobs")

plt.show()
#Biểu đồ scatter plot hiển thị dữ liệu được tạo bằng hàm make_blobs với 2 cụm dữ liệu (clusters).
#Mỗi cụm được tô màu khác nhau để phân biệt.
#Biểu đồ này giúp chúng ta hiểu cách dữ liệu được phân chia thành các cụm và sẽ được sử dụng để huấn luyện mô hình SVM.