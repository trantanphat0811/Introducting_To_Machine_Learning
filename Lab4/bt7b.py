#Duong cong log10 (P(x))
import numpy as np
import matplotlib.pyplot as plt
#Tạo figure và axis
fig=plt.figure()
ax=fig.add_subplot(111)#Tạo subplot 
#Vẽ đồ thị
ax.plot(np.linspace(0.01,1),np.log10(np.linspace(0.01,1)))
ax.set_xlabel("Bieu Dien Ham P(x)")
ax.set_ylabel("log10(P(x))")
#Hiển thị biểu đồ
plt.show() 

#Hàm log10(P(x))là một hàm đơn điệu tăng
#Khi P(X) gần 0, log10(P(x)) giảm mạnh về âm vô cực
#Đồ thị log10 có độ dốc nhỏ hơn log2 vì cơ số thay đổi chậm hơn
#Đồ thị giúp ta thấy rằng càng gần 0, giá trị log càng giảm mạnh và thể hiện rõ cách xác suất ảnh hưởng đến logarit 