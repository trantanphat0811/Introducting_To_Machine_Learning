import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("/Users/trantanphat/Documents/Python/NMHM/Lab5/Iris.csv")

# Lấy cột SepalLengthCm và SepalWidthCm
x = df["SepalLengthCm"]
y = df["SepalWidthCm"]

# Phân loại theo loài
setosa = df[df["Species"] == "Iris-setosa"]
versicolor = df[df["Species"] == "Iris-versicolor"]
virginica = df[df["Species"] == "Iris-virginica"]

# Tạo biểu đồ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(setosa["SepalLengthCm"], setosa["SepalWidthCm"], marker='+', color='green', label="Setosa")
plt.scatter(versicolor["SepalLengthCm"], versicolor["SepalWidthCm"], marker='_', color='red', label="Versicolor")
plt.scatter(virginica["SepalLengthCm"], virginica["SepalWidthCm"], marker='o', color='blue', label="Virginica")

# Thêm nhãn trục và tiêu đề
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset: Sepal Length vs Sepal Width")
plt.legend()
plt.show()
#Biểu đồ scatter plot hiển thị ba nhóm điểm tương ứng với ba loài Iris-setosa, Iris-versicolor, và Iris-virginica dựa trên Sepal Length và Sepal Width.
#Mỗi loài được tô màu khác nhau để phân biệt.
#Trực quan hóa dữ liệu Iris dataset bằng scatter plot, giúp chúng ta quan sát sự phân bố của ba loài hoa dựa trên hai đặc trưng Sepal Length và Sepal Width.







