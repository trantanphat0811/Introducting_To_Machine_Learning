import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product

#đặt tham số
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

#xác định hạt nhân
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

#load tải hình ảnh
image = tf.io.read_file('girl3.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

#vẽ hình ảnh đã được tích chập
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale image')
plt.savefig('original_image.png')

#định dạng lại hình ảnh
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

#lớp tích chập
conv_fn = tf.nn.conv2d

image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1,  # or (1, 1)
    padding='SAME',
)

plt.figure(figsize=(15, 5))

#vẽ hình ảnh đã được tích chập
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Convolution')

#lớp kích hoạt
relu_fn = tf.nn.relu
#phát hiện ảnh
image_detect = relu_fn(image_filter)

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect))
#định dạng lại để vẽ độ chi
plt.axis('off')
plt.title('Activation')

#lớp gộp
pool = tf.nn.pool
image_condense = pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pooling')

plt.savefig('cnn_processing.png')