from sklearn import datasets, tree
import numpy as np #tính x
import matplotlib.pyplot as plt #vẽ đồ thị
from scipy import stats #tính phân phối nhị phân
x=np.arange(50)
plt.plot(x,-np.log(stats.binom.pmf(x,50,0.5)),label='Binom(50,0.5)')
plt.title('Information Content')
plt.xlabel('Value')
plt.ylabel('Information')
plt.plot(x,-np.log(stats.binom.pmf(x,50,0.8)),label='Binom(50,0.8)')
plt.show()

#Trục hoành x là giá trị của biến ngẫu nhiên, trục hoành y là thông tin của giá trị đó.
#Hàm lượng thông tin phản ánh mức độ ngạc nhiên của 1 giá trị 
#Sự kiện càng ít,thông tin mang lại càng nhiều
#Biểu đồ giúp so sánh trực quan 2 phân phối nhị phân có xác suất khác nhau