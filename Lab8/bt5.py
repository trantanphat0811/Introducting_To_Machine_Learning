# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import random
from itertools import product

# Phần 1: KMeans Clustering (Thực hành 04)
# Load file data.csv
dataset = pd.read_csv('data.csv')
print("Dataset from data.csv:")
print(dataset.head())
print(dataset)

n = 1000
dataset1 = np.zeros(shape=(n, 2))
li = 0
for i in range(n):
    j = random.randint(0, 4999)  # Sửa từ randint thành random.randint
    dataset1[li, 0] = dataset.iloc[j, 0]  # Sửa cú pháp numpy array
    dataset1[li, 1] = dataset.iloc[j, 1]
    li += 1

# Siêu tham số
k = 6
kmeans = KMeans(n_clusters=k)
# Huấn luyện k-means
kmeans.fit(dataset1)
gcenters = kmeans.cluster_centers_
print("The geometric centers or centroids:")
print(gcenters)

# Vẽ điểm nhãn
labels = kmeans.labels_
colors = ['blue', 'red', 'green', 'black', 'yellow', 'brown', 'orange']
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

# Kiểm tra tập dữ liệu với dự đoán
x_test = [[40.0, 67], [20.0, 61], [90.0, 90],
          [50.0, 54], [20.0, 80], [90.0, 60]]
prediction = kmeans.predict(x_test)
print("The predictions:")
print(prediction)

# Phần 2: CNN (Bài toán CNN)
# Đặt tham số
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# Xác định hạt nhân
kernel = tf.constant([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Load và xử lý ảnh
image = tf.io.read_file('girl3.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# Vẽ ảnh gốc
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale image')
plt.savefig('original_image.png')

# Định dạng lại ảnh
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# Lớp tích chập
conv_fn = tf.nn.conv2d
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)

# Vẽ ảnh sau tích chập
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Convolution')

# Lớp kích hoạt
relu_fn = tf.nn.relu
image_detect = relu_fn(image_filter)

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Activation')

# Lớp gộp
pool = tf.nn.pool
image_condense = pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME'
)

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pooling')

plt.savefig('cnn_processing.png')