import matplotlib.pyplot as plt

x = [6, 5, 10, 4, 3, 11, 14, 6, 9, 12, 9, 10, 9, 7]
y = [24, 19, 24, 17, 16, 25, 24, 22, 21, 21, 15, 19, 20, 22]

from sklearn.cluster import KMeans

data = list(zip(x, y))
inertias = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 15), inertias, marker='o')
plt.title('Phương pháp ELBOW')
plt.xlabel('số cụm - clusters')
plt.ylabel('Inertia')
plt.show()