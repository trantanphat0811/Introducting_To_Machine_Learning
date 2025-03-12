from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix, solvers

np.random.seed(22)

# Khởi tạo dữ liệu
means = [[2, 2], [4, 2]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T  # (2, N)
X1 = np.random.multivariate_normal(means[1], cov, N).T  # (2, N)
X = np.concatenate((X0, X1), axis=1)  # (2, 2N)

# Nhãn
y = np.concatenate((np.ones((1, N)), -np.ones((1, N))), axis=1)  # (1, 2N)

# Thiết lập bài toán tối ưu bậc hai
V = np.concatenate((X0, -X1), axis=1)  # (2, 2N)
K = matrix(V.T @ V)  # Ma trận Gram (2N, 2N)
p = matrix(-np.ones((2 * N, 1)))
G = matrix(-np.eye(2 * N))  # Ràng buộc lambda >= 0
h = matrix(np.zeros((2 * N, 1)))
A = matrix(y, (1, 2 * N), 'd')
b = matrix(0.0)

solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])  # (2N, 1)
print('lambda = ')
print(l.T)

# Chọn các support vectors
epsilon = 1e-6
S = np.where(l > epsilon)[0]
VS = V[:, S]  # (2, |S|)
XS = X[:, S]  # (2, |S|)
yS = y[:, S]  # (1, |S|)
lS = l[S]  # (|S|, 1)

# Tính toán w và b
w = VS @ lS  # (2, 1)
b = np.mean(yS.T - XS.T @ w)  # Trung bình phần bù

print('w =', w.T)
print('b =', b)

# Chương trình này sử dụng phương pháp SVM đối ngẫu để tìm siêu phẳng tối ưu mà không cần trực tiếp tính
# w va b ban đầu. Thay vào đó, nó giải bài toán tối ưu bậc hai để tìm hệ số 𝜆 từ đó xác định support vectors và suy ra
# w, b. 
