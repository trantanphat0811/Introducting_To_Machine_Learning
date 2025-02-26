import numpy as np
import matplotlib.pyplot as plt

# Tạo figure và axis
fig = plt.figure()
ax = fig.add_subplot(111)

# Dữ liệu đầu vào tránh giá trị 0
x = np.linspace(0.01, 1, num=100)
y = np.log2(x)

# Vẽ đồ thị
ax.plot(x, y, label="log2(P(x))")

# Gán nhãn cho trục
ax.set_xlabel('P(x)')
ax.set_ylabel('log2(P(x))')

# Hiển thị chú thích (legend)
ax.legend()

# Hiển thị đồ thị
plt.show()

# Khi xác suất gần 0 → logarit có giá trị âm rất lớn.
# Khi xác suất gần 1 → logarit tiến dần về 0.
# Kết quả này phản ánh cách logarit xử lý xác suất trong nhiều lĩnh vực như lý thuyết thông tin, học máy, và xử lý dữ liệu
