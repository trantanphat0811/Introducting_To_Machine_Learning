from sklearn.datasets import load_breast_cancer 
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Tải dữ liệu ung thư vú
cancer = load_breast_cancer()
X = cancer.data[:, :2]  # Chỉ lấy 2 đặc trưng đầu tiên
y = cancer.target

# Huấn luyện mô hình SVM tuyến tính
svm = SVC(kernel='linear', C=1)
svm.fit(X, y)

# Vẽ ranh giới quyết định
disp = DecisionBoundaryDisplay.from_estimator(
    svm, X, response_method='predict', cmap=plt.cm.Spectral, alpha=0.8
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')

# Đặt tiêu đề cho trục
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])

plt.show()
#Mô hình SVM học được một đường thẳng (vì kernel là linear) để phân chia hai nhóm dữ liệu (lành tính và ác tính).
#Ranh giới quyết định được hiển thị bằng màu sắc khác nhau, Dữ liệu thực tế được vẽ lên biểu đồ giúp kiểm tra xem SVM phân loại tốt hay không
#Vì dữ liệu ban đầu có 30 đặc trưng, chỉ dùng 2 đặc trưng đầu tiên có thể không đủ để phân loại chính xác.