import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Dữ liệu huấn luyện
e1 = [2, 1, 2, 1, 0, 2] #vector đặc trưng đại diện mẫu huấn luyện
e2 = [2, 1, 1, 0, 2, 1]
e3 = [2, 0, 1, 1, 2, 1]
e4 = [1, 0, 1, 1, 1, 2]
e5 = [2, 1, 1, 0, 1, 1]

train_data = np.array([e1, e2, e3, e4, e5])  # Ma trận chứa dữ liệu huấn luyện
class_ = np.array(['Y', 'Y', 'Y', 'N', 'N'])  # Nhãn tương ứng train_data

# Dữ liệu kiểm tra
e6 = np.array([[1, 1, 1, 1, 1, 2]]) #Mẫu kiểm tra mô hình

# Khởi tạo và huấn luyện mô hình Naive Bayes
ml = MultinomialNB(alpha=1)
ml.fit(train_data, class_)

# Dự đoán xác suất và nhãn cho e6
y_pred_proba = ml.predict_proba(e6) #Trả xác suất mẫu
y_pred = ml.predict(e6)[0]

# In kết quả
print('Probability of e6 :', y_pred_proba) #Trả xác suất mẫu 
print('Predicting class of e6 :', y_pred) # Dự đoán xác suất cao nhất

# Xác suất e6 thuộc lớp Y là 46,23% và lớp N là 53,77% 
#Mô hình được huấn luyện thành công với naive bayes, kết quả phù hợp với mô hình 
#Xác suất e6 thuộc lớp Y
#Dự đoán nhãn e6
