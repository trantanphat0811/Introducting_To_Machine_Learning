import numpy as np
def gini_index(groups, classes): # Tổng số phần tử 
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups: #Tính chỉ số gini cho từng nhóm 
        size = float(len(group))
        if size == 0:
            continue #bỏ qua nhóm rỗng
        score = 0.0
        for class_val in classes: #tính độ thuần khiết của nhóm
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances) #tính gini cho nhóm
    return gini
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1])) #kiểm thử kết quả
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

#Khi nhóm chứa nhiều lớp lẫn lộn, chỉ số Gini cao (0.5).
#Khi nhóm thuần nhất, chỉ số Gini thấp (0.0).
#Hàm này rất quan trọng trong cây quyết định, nơi ta chọn cách chia dữ liệu sao cho chỉ số Gini giảm.