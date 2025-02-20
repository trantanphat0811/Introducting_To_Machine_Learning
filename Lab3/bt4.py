import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Đọc dữ liệu từ file
persons = []
with open("/Users/trantanphat/Documents/Python/NMHM/Lab3/dochieucao.txt", encoding="utf-8") as fh:
    for line in fh:
        parts = line.strip().split()
        if len(parts) < 5:  # Kiểm tra nếu dòng không đủ dữ liệu
            print("Dòng lỗi:", parts)
            continue  
        
        name = " ".join(parts[:-4])  # Ghép các phần đầu làm tên
        age, height, weight, gender = parts[-4:]  # Lấy 4 phần cuối làm dữ liệu số
        persons.append([name, int(age), int(height), int(weight), gender])

# Xử lý dữ liệu theo giới tính
genders = ["male", "female"]
firstnames = {gender: [] for gender in genders}
heights = {gender: [] for gender in genders}

# Lấy dữ liệu theo giới tính
for person in persons:
    name, age, height, weight, gender = person
    if gender in genders:
        firstnames[gender].append(name)
        heights[gender].append(height)  # Dữ liệu đã chuyển thành số nguyên

# Kiểm tra dữ liệu sau khi xử lý
for gender in genders:
    print(f"{gender}:")
    print("Tên:", firstnames[gender][:5])
    print("Chiều cao:", heights[gender][:5])

# Định nghĩa lớp Feature
class Feature:
    def __init__(self, data, name=None, bin_width=None):
        self.name = name
        self.bin_width = bin_width
        if not data:  # Kiểm tra nếu dữ liệu rỗng
            print(f"Lỗi: Không có dữ liệu cho {name}")
            self.freq_dict = {}
            self.freq_sum = 0
            return
        
        if bin_width:
            self.min, self.max = min(data), max(data)
            bins = np.arange((self.min // bin_width) * bin_width,
                             (self.max // bin_width) * bin_width + bin_width,
                             bin_width)
            freq, bins = np.histogram(data, bins)
            self.freq_dict = dict(zip(bins, freq))
            self.freq_sum = sum(freq)
        else:
            self.freq_dict = dict(Counter(data))
            self.freq_sum = sum(self.freq_dict.values())

    def frequency(self, value):
        if self.bin_width:
            value = (value // self.bin_width) * self.bin_width
            return self.freq_dict.get(value, 0)
        return self.freq_dict.get(value, 0)

# Tạo Feature cho chiều cao
fts = {}
for gender in genders:
    fts[gender] = Feature(heights[gender], name=gender, bin_width=5)
    print(f"{gender}: {fts[gender].freq_dict}")

# Vẽ biểu đồ phân bố chiều cao
for gender in genders:
    frequencies = list(fts[gender].freq_dict.items())
    frequencies.sort(key=lambda x: x[1])
    X, Y = zip(*frequencies) if frequencies else ([], [])
    color = "blue" if gender == "male" else "red"
    bar_width = 4 if gender == "male" else 3
    plt.bar(X, Y, bar_width, color=color, alpha=0.75, label=gender)

plt.legend(loc='upper right')
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Số lượng")
plt.title("Phân bố chiều cao theo giới tính")
plt.show()

# Định nghĩa lớp Naive Bayes
class NBclass:
    def __init__(self, name, *features):
        self.features = features
        self.name = name

    def probability_value_given_feature(self, feature_value, feature):
        if feature.freq_sum == 0:
            return 0
        return feature.frequency(feature_value) / feature.freq_sum

# Tạo đối tượng NBclass cho từng giới tính
cls = {gender: NBclass(gender, fts[gender]) for gender in genders}

# Định nghĩa bộ phân loại
class Classifier:
    def __init__(self, *nbclasses):
        self.nbclasses = nbclasses

    def prob(self, *d, best_only=True):
        probability_list = []
        for nbclass in self.nbclasses:
            ftrs = nbclass.features
            prob = 1
            for i in range(len(ftrs)):
                prob *= nbclass.probability_value_given_feature(d[i], ftrs[i])
            probability_list.append((prob, nbclass.name))

        # Chuẩn hóa xác suất nếu tổng bằng 0
        prob_values = [f[0] for f in probability_list]
        prob_sum = sum(prob_values)
        if prob_sum == 0:
            number_classes = len(self.nbclasses)
            probability_list = [(1 / number_classes, p[1]) for p in probability_list]
        else:
            probability_list = [(p[0] / prob_sum, p[1]) for p in probability_list]

        return max(probability_list) if best_only else probability_list

# Sử dụng bộ phân loại
c = Classifier(cls["male"], cls["female"])

# In xác suất chiều cao thuộc về giới tính nào
for i in range(130, 220, 5):
    print(i, c.prob(i, best_only=False))

