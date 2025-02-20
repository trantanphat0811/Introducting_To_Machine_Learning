#import thư viện và khởi tạo dữ liệu
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
X = np.array([0.20, 0.35, 1.00, 1.30, 1.35, 1.45, 1.50, 2.00, 2.10, 2.25,
              3.00, 3.05, 3.10, 3.50, 3.60, 4.05, 4.10, 5.85, 5.90])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1])

#Chuẩn bị dữ liệu
X = X.reshape(1, -1) 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)  

#định nghĩa các hàm cần thiết 
def sigmoid(s): #kích hoạt hàm sigmoid
    return 1 / (1 + np.exp(-s))

#Kích hoạt hàm hồi quy logistic
def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-1]) < tol:
                    return w
            w.append(w_new)
    return w

#huấn luyện mô hình
eta = 0.05
d = X.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(X, y, w_init, eta)

#In kết quả
print("Final weights:", w[-1].flatten())
probs = sigmoid(np.dot(w[-1].T, X))
print("Predicted probabilities:", probs.flatten())


#Trực quan hoá dữ liệu và nhãn
X0 = X[1, y == 0]
y0 = y[y == 0]
X1 = X[1, y == 1]
y1 = y[y == 1]

plt.plot(X0, y0, 'ro', markersize=8, label="Class 0")
plt.plot(X1, y1, 'bs', markersize=8, label="Class 1")


#Trực quan hoá đường sigmoid
xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0 / w1
yy = sigmoid(w0 + w1 * xx)
plt.plot(xx, yy, 'g-', linewidth=2, label="Sigmoid curve")
plt.plot(threshold, 0.5, 'y^', markersize=8, label="Threshold")

#Hoàm thiện biểu diễn
plt.axis([-1, 7, -0.1, 1.1])
plt.xlabel("Studying hours")
plt.ylabel("Predicted probability")
plt.legend()
plt.show()
