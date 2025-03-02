from math import log2
import matplotlib.pyplot as plt

# Danh sách xác suất (loại bỏ 0 để tránh lỗi log2(0))
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Tính lượng thông tin I(p) = -log2(p), tránh lỗi bằng cách bỏ qua p=0
info = [-log2(p) if p > 0 else float('inf') for p in probs]

# Vẽ đồ thị
plt.plot(probs, info, marker='o', linestyle='-')
plt.title('Probability vs Information')
plt.xlabel('Probability')
plt.ylabel('Information (bits)')
plt.grid(True)
plt.show()
#Xác suất cao (gần 1.0) ⟶ Lượng thông tin gần 0 
#Xác suất thấp (gần 0.1) ⟶ Lượng thông tin cao
#Đường cong giảm dần: Lượng thông tin giảm khi xác suất tăng, đúng theo trực giác và công thức.
#minh họa trực quan cách lượng thông tin thay đổi theo xác suất!
