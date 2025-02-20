import numpy as np
import matplotlib.pyplot as plt

#kích hoạt hàm sigmoid nhằm chuyển đầu ra một giá trị xác suất từ 0 đến 1
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

#Hàm tính log_likelihood để đánh giá mức độ dữ liệu
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll

#Hàm huấn luyện hồi quy
def logistic_regression(features, target, num_steps,learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        weights = np.zeros(features.shape[1])
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient 
    if step % 10000 == 0: 
        print(log_likelihood(features, target, weights))
        
        return weights 
    
#Tạo dữ liệu mô phỏng cho dữ liệu 
np.random.seed(10)
num_observations = 10000
x1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]],
num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.5], [0.5, 1]],
num_observations)
simulated_separableish_features = np.vstack((x1,
x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
np.ones(num_observations)))

#Trực quan hoá dữ liệu
plt.figure(figsize=(10, 8))
plt.scatter(simulated_separableish_features[:,
0],simulated_separableish_features[:, 1],c=simulated_labels,alpha=0.3,)
plt.show()

#scatter plot của 2 nhóm dữ liệu với nhóm dữ liệu 1 và 2
#Trục x, y thể hiện đặc trưng dữ liệu
#Biểu đồ trên cho thấy mức độ chồng lấn rõ ràng giữa 2 nhóm dữ liệu 
# bằng 1 bài toán nhị phân
