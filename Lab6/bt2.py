import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=800, n_features=2,centers=4, random_state=23)
fig= plt.figure(0)
plt.grid (True)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#Đồ thị hiển thị 800 điểm dữ liệu với 4 cụm khác nhau
# Mỗi cụm có màu sắc khác nhau với khoảng cách đều nhau
# Sử dụng hàm make_blobs từ thư viện sklearn.datasets để tạo dữ liệu ngẫu nhiên
