from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from graphviz import Source

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Khởi tạo và huấn luyện mô hình cây hồi quy
tree_reg = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
tree_reg.fit(X, y)

# Xuất đồ thị cây quyết định dưới dạng DOT format
dot_data = export_graphviz(
    tree_reg,
    out_file=None,  # Xuất ra chuỗi DOT thay vì file
    feature_names=diabetes.feature_names,
    rounded=True,
    filled=True
)

# Hiển thị cây quyết định bằng graphviz
graph = Source(dot_data)
graph.render("diabetes_tree", format="png", cleanup=True)  # Lưu ra file PNG
graph.view()  # Tự động mở file ảnh cây


# Mỗi node trên cây hiển thị:
# Điều kiện phân nhánh (theo đặc trưng nào và giá trị cụ thể).
# mse: sai số bình phương trung bình của node đó.
# samples: số lượng mẫu dữ liệu tại node.
# value: giá trị dự đoán trung bình của node đó (giá trị đầu ra của mô hình tại node lá).
# Cây sẽ phân chia dữ liệu dựa trên các đặc trưng đầu vào để tìm ra cách dự đoán tốt nhất giá trị mục tiêu (mức độ bệnh).
