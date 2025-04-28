import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

# Cấu hình hiển thị
plt.rc('figure', autolayout=True)

# Khởi tạo bộ lọc Laplacian (Edge Detection)
kernel = tf.constant([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], tf.float32)

# Đọc ảnh, chuyển sang grayscale và resize
image = tf.io.read_file('girl3.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=(300, 300))
img = tf.squeeze(image).numpy()  # Chuyển về numpy array để hiển thị

# Hiển thị ảnh gốc
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')  # Giữ nguyên cmap cho grayscale
plt.axis('off')
plt.title('Original Image')
plt.show()

# Chuẩn hóa ảnh về dạng [0,1] (float32)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)  # Thêm batch dimension

# Định dạng kernel cho phù hợp với ảnh grayscale (1 kênh)
kernel = tf.reshape(kernel, [3, 3, 1, 1])

# Tích chập ảnh với kernel (Convolution)
image_filter = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')

# Hiển thị kết quả tích chập
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter))  # Sửa lỗi tf.squeez() thành tf.squeeze()
plt.axis('off')
plt.title('Convolution')

# Áp dụng hàm kích hoạt ReLU
image_detect = tf.nn.relu(image_filter)
plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Activation')

# Áp dụng Max Pooling để giảm kích thước ảnh
image_condense = tf.nn.max_pool2d(image_detect, ksize=2, strides=2, padding='SAME')

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pooling')
plt.show()

#Ảnh gốc: Hiển thị ảnh ban đầu.
#Convolution: Làm nổi bật biên cạnh bằng bộ lọc Laplacian.
#ReLU: Loại bỏ giá trị âm, giữ lại các biên cạnh dương.
#Max Pooling: Giảm kích thước ảnh, giữ lại các đặc trưng quan trọng.
#Kết quả cuối cùng là một ảnh nhỏ hơn, với các biên cạnh rõ ràng hơn, phù hợp cho các bước xử lý tiếp theo như nhận diện hoặc phân loại.