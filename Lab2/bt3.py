import numpy as np
import matplotlib.pyplot as plt
def sigmoid(scores): #Kích hoạt hàm logistic
    return 1 / (1 + np.exp(-scores))

#Tạo các danh sách giá trị cho dữ liệu
xs = [scores / 10.0 for scores in range(-50, 50)]

#Vẽ đồ thị đường cong sigmoid
plt.plot(xs,[sigmoid(scores) for scores in xs],'.',label='Logistic Regression')
plt.title("Duong Cong Chu S - sigmoid") 
plt.show()