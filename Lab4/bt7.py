
#Duong cong log2 (P(x))
import numpy as np
import matplotlib.pyplot as plt
#Tạo figure và axis
fig=plt.figure()

#Vẽ đồ thị
ax=fig.add_subplot(111) #Tạo subplot 
ax.plot(np.linspace(0.01,1),np.log2(np.linspace(0.01,1)))
ax.set_xlabel("Ham P(x)")
ax.set_ylabel("log2(P(x))") 

#Hiển thị biểu đồ
plt.show()


#Khi P(x) gần 0, log2(P(x)) giảm mạnh về âm vô cực 
#Khi P(x) = 1 .  log2(1) =0
#Đồ thị giúp ta thấy rằng xác suất càng nhỏ, lượng thông tin càng lớn