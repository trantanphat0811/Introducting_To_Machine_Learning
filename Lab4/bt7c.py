#Duong cong ln (P(x))
import math
import numpy as np
import matplotlib.pyplot as plt

#Tạo figure và axis
fig=plt.figure()

#Vẽ đồ thị
ax=fig.add_subplot(111) #Tạo subplot
ax.plot(np.linspace(0.01,1),np.log(np.linspace(0.01,1)))
ax.set_xlabel("Bieu Dien Ham P(x)")
ax.set_ylabel("ln(P(x))")

#Hiển thị đồ thị
plt.show()

#Hàm ln(P(x))là một hàm đơn điệu tăng
#Khi P(X) gần 0, ln(P(x)) giảm mạnh về âm vô cực
#Khi P(X) = 1 . ln(1) =0
#Hàm ln(x) thay đổi chậm hơn so với hàm log10(x) và log2(x)
