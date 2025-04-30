import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#load file VI.csv
df = pd.read_csv('VI.csv')
print(df.head())

#KNN-nhánh phân loại
X = df.loc[:, 'broke':'shouted']
Y = df.loc[:, 'class']

#mô hình huấn luyện
knn = KNeighborsClassifier()
knn.fit(X, Y)

X_DL = [[9, 0, 9, 0]]
prediction = knn.predict(X_DL)
print("The prediction is:", str(prediction).strip('[]'))

#quy tắc từ vùng của tập dữ liệu
df = pd.read_csv('VI.csv')

#biểu đồ- mối quan hệ của từng đặc điểm với từng lớp
figure, (sub1, sub2, sub3, sub4) = plt.subplots(4, sharex=True, sharey=True)
plt.suptitle('k-nearest neighbors')

#Thực hành 02:
plt.xlabel('Feature')
plt.ylabel('Class')

X = df.loc[:, 'broke']
Y = df.loc[:, 'class']
sub1.scatter(X, Y, color='blue', label='broke')
sub1.legend(loc=4, prop={'size': 5})
sub1.set_title('Polysemy')

X = df.loc[:, 'road']
Y = df.loc[:, 'class']
sub2.scatter(X, Y, color='green', label='road')
sub2.legend(loc=4, prop={'size': 5})

X = df.loc[:, 'stopped']
Y = df.loc[:, 'class']
sub3.scatter(X, Y, color='red', label='stopped')
sub3.legend(loc=4, prop={'size': 5})

X = df.loc[:, 'shouted']
Y = df.loc[:, 'class']
sub4.scatter(X, Y, color='black', label='shouted')
sub4.legend(loc=4, prop={'size': 5})

figure.subplots_adjust(hspace=0)
plt.savefig('knn_scatter_plots.png')