from math import log2
import matplotlib.pyplot as plt

# Hàm tính entropy
def entropy(ents, eps=1e-15):
    return -sum([p * log2(p + eps) for p in ents if p > 0])  # Tránh log(0)

# Danh sách xác suất P(X=1), còn lại là P(X=0)
probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
dists = [[p, 1.0 - p] for p in probs]

# Tính entropy cho từng phân phối
ents = [entropy(d) for d in dists]

# Vẽ đồ thị
plt.plot(probs, ents, marker='o', linestyle='-')
plt.title('Probability vs Entropy')
plt.xticks(probs, [str(d) for d in dists], rotation=45)  # Xoay nhãn trục x cho dễ đọc
plt.xlabel('Probability (P)')
plt.ylabel('Entropy (bits)')
plt.grid(True)
plt.show()

# Entropy đo độ không chắc chắn của một hệ thống.
# Entropy lớn nhất khi xác suất là 0.5, và nhỏ nhất khi xác suất gần 0 hoặc 1.
# Kết quả này phản ánh cách entropy xử lý xác suất trong nhiều lĩnh vực như lý thuyết thông tin, học máy, và xử lý dữ liệu