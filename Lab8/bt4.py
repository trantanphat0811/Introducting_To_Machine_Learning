from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import random
import numpy as np

#file data.csv
dataset = pd.read_csv('data.csv')
print(dataset.head())
print(dataset)

n = 1000
dataset1 = np.zeros(shape=(n, 2))
li = 0
for i in range(n):
    j = randint(0, 4999)
    dataset1[li][0] = dataset.iloc[j, 0]
    dataset1[li][1] = dataset.iloc[j, 1]
    li += 1

#Thực hành 04:
#siêu tham số
k = 6
kmeans = KMeans(n_clusters=k)
#huấn luyện k-means
kmeans.fit(dataset1)
gcenters = kmeans.cluster_centers_  #các điểm giữa trung tâm
print("The geometric centers or centroids:")
print(gcenters)

#vẽ điểm nhãn
labels = kmeans.labels_
colors = ['blue', 'red', 'green', 'black', 'yellow', 'brown', 'orange']
#kiểm tra: điểm dữ liệu va cum
y = 0
for x in labels:
    plt.scatter(dataset1[y, 0], dataset1[y, 1], color=colors[x])
    y += 1
for x in range(k):
    lines = plt.plot(gcenters[x, 0], gcenters[x, 1], 'kx')

title = 'No of clusters (k) = {}'.format(k)
plt.title(title)
plt.xlabel('Distance')
plt.ylabel('Location')
plt.savefig('kmeans_clusters.png')

#kiểm tra tập dữ liệu với dự đoán
x_test = [[40.0, 67], [20.0, 61], [90.0, 90],
          [50.0, 54], [20.0, 80], [90.0, 60]]
prediction = kmeans.predict(x_test)
print("The predictions:")
print(prediction)