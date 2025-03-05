from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
import graphviz

# Load dữ liệu Iris
iris = load_iris()
# Lấy đặc trưng chiều dài và chiều rộng cánh hoa
X = iris.data[:, 2:4]
y = iris.target

# Huấn luyện mô hình cây quyết định
tree_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
tree_clf.fit(X, y)

# Xuất cây quyết định dưới dạng chuỗi DOT
dot_data = export_graphviz(
    tree_clf,
    out_file=None,  # Xuất trực tiếp ra chuỗi, không ghi file
    feature_names=iris.feature_names[2:4],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# Hiển thị cây quyết định
graph = Source(dot_data)
graph.render('iris_tree', format='png', cleanup=True)  # Xuất ra file ảnh PNG nếu muốn
graph.view()  # Mở ảnh hoặc hiển thị trực tiếp nếu môi trường hỗ trợ


#Thu được hình ảnh trực quan của cây quyết định phân loại hoa iris dựa trên chiều dài và chiều rộng của cây
#Mỗi nút trên cây hiển thị:
