import numpy as np

# Hàm chia dữ liệu thành hai nhóm dựa trên chỉ số và giá trị
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Hàm tính chỉ số Gini để đo độ hỗn hợp của nhóm dữ liệu
def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))  # Tổng số phần tử
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue  # Tránh chia cho 0
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size  # Tỷ lệ class_val trong nhóm
            score += p * p
        gini += (1.0 - score) * (size / n_instances)  # Tính Gini với trọng số theo kích thước nhóm
    return gini  # Thêm return để trả về giá trị Gini

# Hàm tìm cách chia dữ liệu tốt nhất
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Lấy danh sách các nhãn lớp duy nhất
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None

    for index in range(len(dataset[0])-1):  # Duyệt qua tất cả các cột (trừ nhãn lớp)
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print(f'X{index+1} < {row[index]:.3f}  -->  Gini = {gini:.3f}')
            if gini < best_score:  # Nếu Gini nhỏ hơn giá trị tốt nhất trước đó
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups

    if best_groups is None:
        return None  # Tránh lỗi khi không tìm thấy cách chia nào tốt hơn

    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Hàm dự đoán lớp đầu ra dựa trên cây quyết định (stump)
def predict(stump, row):
    if row[stump['index']] < stump['value']:
        return stump['left']
    else:
        return stump['right']

# Dữ liệu thử nghiệm (2 đặc trưng, 1 nhãn lớp)
dataset =[[2.771244718,1.784783929,0],
[1.728571309,1.169761413,0],
[3.678319846,2.81281357,0],
[3.961043357,2.61995032,0],
[2.999208922,2.209014212,0],
[7.497545867,3.162953546,1],
[9.00220326,3.339047188,1],
[7.444542326,0.476683375,1],
[10.12493903,3.234550982,1],
[6.642287351,3.319983761,1]]

split = get_split(dataset)
# chương trình tìm ra đặc trưng tốt nhất để chia dữ liệu.
#Chỉ số Gini được sử dụng để đánh giá mức độ hỗn hợp của các nhóm.
#Tìm kiếm toàn bộ giá trị khả dĩ để tìm điểm chia tối ưu.
#Tìm ra điểm chia tốt nhất giúp xây dựng cây quyết định tối ưu.
#Đây là bước quan trọng trong thuật toán Decision Tree, được sử dụng trong các bài toán phân loại như ID3, C4.5, CART.