import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# Tạo dữ liệu
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 2, 8.8, 7.5])

# Hiển thị dữ liệu ban đầu
plt.scatter(X, y, color='b', label="Dữ liệu")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Chuẩn bị dữ liệu huấn luyện
training_X = np.vstack((X, y)).T  # Chuyển thành ma trận (12, 2)
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]  # Nhãn của dữ liệu

# Huấn luyện SVM với kernel tuyến tính
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(training_X, training_y)

# Lấy trọng số w và tính hệ số góc đường quyết định
w = clf.coef_[0]  # Trọng số của mô hình
a = -w[0] / w[1]  # Hệ số góc

# Vẽ ranh giới quyết định
XX = np.linspace(0, 13, 100)  # Tạo dãy điểm trên trục x
yy = a * XX - clf.intercept_[0] / w[1]  # Tính giá trị y tương ứng

plt.plot(XX, yy, 'k-', label="Ranh giới quyết định")  # Vẽ đường quyết định
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y, cmap=plt.cm.Spectral, edgecolors='k', label="Dữ liệu huấn luyện")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

#Biểu đồ 1 (Scatter Plot): Hiển thị dữ liệu gốc với các điểm màu xanh.
#Biểu đồ 2 (SVM Decision Boundary)
#Các điểm dữ liệu được tô màu theo lớp (0 và 1).
#Đường thẳng màu đen biểu diễn ranh giới quyết định giữa hai lớp.
#Giúp trực quan hóa cách SVM phân tách dữ liệu.
