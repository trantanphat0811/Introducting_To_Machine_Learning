import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Tạo dữ liệu mẫu
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(80, 1) - 100, axis=0)  # Giá trị từ -100 đến 100
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))  # Thêm nhiễu

# Khởi tạo và huấn luyện mô hình
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Tạo dữ liệu kiểm tra
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="Dữ liệu mẫu")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("Giá trị đầu vào (X)")
plt.ylabel("Giá trị dự đoán (y)")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


#Dữ liệu mẫu có dạng sóng sin nhưng bị nhiễu, thể hiện bằng các chấm màu cam.
#Đường dự đoán với max_depth=2:
#Cây rất nông nên chỉ chia thành vài đoạn bậc thang lớn.
#Dự đoán khá thô, không bám sát dữ liệu thật.
#Đường dự đoán với max_depth=5:
#Cây sâu hơn nên có nhiều phân chia, mô phỏng tốt hơn hình dạng sóng sin.
#Dự đoán mượt mà hơn và bám sát hơn dữ liệu mẫu.
