import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Sinh dữ liệu
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))

# Khởi tạo và huấn luyện các mô hình Decision Tree Regression
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# Tạo dữ liệu test
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
s = 20
plt.scatter(y[:, 0], y[:, 1], c="navy", s=s, edgecolor="black", label="Original data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s, edgecolor="black", label="max_depth=2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s, edgecolor="black", label="max_depth=5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="green", s=s, edgecolor="black", label="max_depth=8")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.xlabel("Target OX")
plt.ylabel("Target OY")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()

#Mô phỏng khả năng của cây hồi quy khi giải quyết bài toán có nhiều đầu ra.
#Biểu đồ trực quan cho thấy mức độ phù hợp của từng mô hình theo độ sâu.
#max_depth=2: mô hình đơn giản, đường dự đoán thô, không bám sát dữ liệu.
#max_depth=5: cải thiện rõ rệt, bám gần dữ liệu hơn.
#max_depth=8: mô hình phức tạp, có khả năng bám sát dữ liệu rất tốt, nhưng cũng dễ bị overfitting (quá khớp với dữ liệu nhiễu).
