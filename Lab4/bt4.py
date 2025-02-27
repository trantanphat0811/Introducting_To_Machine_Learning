def predict(node, row):
# Dự đoán lớp đầu ra dựa trên cây quyết định
    if row[node['index']] < node['value']:
        return node['left'] if not isinstance(node['left'], dict) else predict(node['left'], row)
    else:
        return node['right'] if not isinstance(node['right'], dict) else predict(node['right'], row)

# Tập dữ liệu thử nghiệm
dataset = [
    [2.771244718, 1.784783929, 0],
    [1.728571309, 1.169761413, 0],
    [3.678319846, 2.81281357, 0],
    [3.961043357, 2.61995032, 0],
    [2.999208922, 2.209014212, 0],
    [7.497545867, 3.162953546, 1],
    [9.00220326, 3.339047188, 1],
    [7.444542326, 0.476683375, 1],
    [10.12493903, 3.234550982, 1],
    [6.642287351, 3.319983761, 1]
]

# Stump (Cây quyết định gốc)
stump = {'index': 0, 'value': 6.642287351, 'left': 0, 'right': 1}

# Dự đoán và in kết quả
for row in dataset:
    prediction = predict(stump, row)
    print(f'Expected={row[-1]}, Got={prediction}')

#Những dòng có X1 < 6.642 được dự đoán là 0 (chính xác).
#Những dòng có X1 >= 6.642 được dự đoán là 1 (chính xác).
#Chương trình dự đoán đúng toàn bộ (100% accuracy) vì tập dữ liệu này phân tách hoàn toàn dựa trên X1.