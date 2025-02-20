import math
from sklearn.naive_bayes import MultinomialNB
import numpy as np
e1 = [2, 1, 2, 2, 2, 2] #Vector đặc trưng đại diện cho mẫu dữ liệu huấn luyện
e2 = [2, 2, 2, 1, 2, 2]
e3 = [2, 2, 2, 2, 2, 2]
e4 = [0, 0, 1, 0, 2, 1]
train_data = np.array([e1, e2, e3, e4]) #Ma trận chứa các mẫu huấn luyện
ket_qua = np.array(['Y', 'Y', 'Y', 'N']) #Nhãn của từng mảng trong train_data
e5 = np.array([[1, 1, 2, 1, 2, 1]]) #Mẫu kiểm tra mô hình
ml = MultinomialNB(alpha=1)
ml.fit(train_data, ket_qua) #quá trình huấn luyện mô hình 
print('Probability of e5 :', ml.predict_proba(e5)) #trả xác suất mẫu 
print('Predicting class of e5 :', str(ml.predict(e5)[0])) #Dự đoán xác suất cao nhất
#Xác suất e5 thuộc lớp Y
#Dự đoán nhãn e5