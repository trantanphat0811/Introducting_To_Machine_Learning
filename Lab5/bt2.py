from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

# Đặt seed để kết quả nhất quán
np.random.seed(22)

# Tạo dữ liệu ngẫu nhiên
means = [[2, 2], [4, 2]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T  # (2, N)
X1 = np.random.multivariate_normal(means[1], cov, N).T  # (2, N)
X = np.concatenate((X0, X1), axis=1)  # (2, 2N)
y = np.concatenate((np.ones((1, N)), -np.ones((1, N))), axis=1)  # (1, 2N)

# Chuyển đổi dữ liệu thành định dạng phù hợp cho sklearn
X_train = X.T  # (2N, 2)
y_train = y.flatten()  # (2N,)

# Huấn luyện SVM với kernel tuyến tính
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Lấy trọng số w và bias b từ mô hình đã huấn luyện
w = clf.coef_.flatten()  # Chuyển về dạng vector 1D
b = clf.intercept_[0]  # Lấy giá trị thực

# In kết quả
print('w =', w)
print('b =', b)

# Tạo dữ liệu phân phối chuẩn từ phân phối đa biến 
# Chuyển đổi dữ liệu phù hợp với sklearn
# Mô hình này có thể được trực quan hóa bằng cách vẽ siêu phẳng và các hỗ trợ vector trên mặt phẳng 2D để dễ dàng hiểu cách SVM phân tách dữ liệu.