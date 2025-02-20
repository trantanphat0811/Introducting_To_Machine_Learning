import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression #Hồi quy logistic từ thư viện sklearn
from sklearn.metrics import classification_report,confusion_matrix

x = np.arange(10).reshape(-1, 1) #Tạo dữ liệu mẫu có x  là một mảng từ 0 đến 9
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) #Nhãn mục tiêu sử dụng mảng nhị phân
x
y
model = LogisticRegression(solver='liblinear', random_state=0) # Huấn luyện mô hình hồi quy
model.fit(x, y)
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y) #Huấn luyện mô hình x và y
model.classes_ #Các mô hình phân loại
model.intercept_ #Hệ số chặn
model.coef_  #Hệ số ứng biến độc lập x

#Dự đoán và đánh giá mô hình
model.predict_proba(x) 
model.predict(x)
model.score(x, y)

confusion_matrix(y, model.predict(x)) #Tạo ra ma trận nhằm biểu diễn số đúng và sai cho từng lớp
cm = confusion_matrix(y, model.predict(x)) 

#Vẽ biểu đồ ma trận
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s')) #Actual 0s -Nhãn thực tế là 0 
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s')) #Actual 1s-Nhãn thực tế là 1
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', 
            va='center', color='red')
plt.show()
print(classification_report(y, model.predict(x)))

#Mô hình dự đoán đúng 3 trường hợp là lớp 0 ,1 mô hình nhầm trường hợp từ 0 thành 1

                           

