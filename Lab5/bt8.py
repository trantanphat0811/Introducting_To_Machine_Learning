import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Tạo dữ liệu dạng vòng tròn
X, y = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=0)

# Tạo giá trị xfit để vẽ đường phân chia
xfit = np.linspace(-1, 3.5, 100)

# Vẽ scatter plot với hai vòng tròn
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring', edgecolors='k')

# Vẽ các đường phân chia và vùng ranh giới
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')  # Vẽ đường thẳng
    plt.fill_between(xfit, yfit - d, yfit + d, color='#AAAAAA', alpha=0.4)  # Tô vùng ranh giới

# Căn chỉnh giới hạn trục
plt.xlim(-1, 3.5)
plt.ylim(-1, 5)

# Hiển thị biểu đồ
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Vòng tròn dữ liệu với các đường phân chia")
plt.show()
#Biểu đồ scatter plot hiển thị dữ liệu vòng tròn được tạo bằng hàm make_circles.
#Mỗi vòng tròn được tô màu khác nhau để phân biệt.
#Đồng thời, biểu đồ cũng vẽ các đường phân chia và vùng ranh giới tương ứng với mỗi đường phân chia.