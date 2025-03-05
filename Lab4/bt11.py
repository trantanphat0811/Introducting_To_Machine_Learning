import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Đọc dữ liệu từ file CSV
df = pd.read_csv('/Users/trantanphat/Documents/Python/NMHM/Lab4/data.csv')

# Mã hóa dữ liệu
nationality_mapping = {"UK": 0, "USA": 1, "N": 2}
go_mapping = {"YES": 1, "NO": 0}
df["Nationality"] = df["Nationality"].map(nationality_mapping)
df["Go"] = df["Go"].map(go_mapping)

# Xác định đặc trưng và nhãn
features = ["Age", "Experience", "Rank", "Nationality"]
X = df[features]
y = df["Go"]

# Khởi tạo và huấn luyện mô hình cây quyết định
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# Xuất cây quyết định thành đồ họa
dot_data = export_graphviz(
    dtree,
    out_file=None,
    feature_names=features,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('my_decision_tree.png')

# Hiển thị ảnh cây quyết định
img = mpimg.imread('my_decision_tree.png')
plt.imshow(img)
plt.axis('off')  # Tắt trục để hình đẹp hơn
plt.show()

#mô hình cây quyết định được huấn luyện để dự đoán liệu 1 người có đi hay không dựa trên "age", "experience", "rank" và "national
#Hiển thị trực quan cây quyết định với màu sắc giúp phân biệt nhánh và kết quả dễ dàng.

