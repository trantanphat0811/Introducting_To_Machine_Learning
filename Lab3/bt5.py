import numpy as np

# Đường dẫn file 
file_path = "/Users/trantanphat/Documents/Python/NMHM/Lab3/chieucao.txt"

def clean_and_prepare_dataset(fname):
    persons = []
    with open(fname, "r", encoding="utf-8") as fh:
        for line in fh:
            person = line.strip().split()
            
            # Kiểm tra số lượng phần tử
            if len(person) < 5:
                print(f"⚠️ Bỏ qua dòng không hợp lệ: {person}")
                continue

            # Lấy dữ liệu
            try:
                firstname = person[0] + " " + person[1]  # Ghép họ + tên
                height = float(person[2])  # Chuyển chiều cao thành số thực
                weight = float(person[3])  # Chuyển cân nặng thành số thực
                gender = person[4].lower()  # Chuyển giới tính về chữ thường

                # Kiểm tra giới tính có hợp lệ không
                if gender not in ["male", "female"]:
                    print(f" Lỗi: Giới tính không hợp lệ ở dòng: {person}")
                    continue
                
                persons.append(((firstname, height, weight), gender))

            except ValueError:
                print(f"Lỗi: Dữ liệu không hợp lệ ở dòng: {person}")

    return persons

# Gọi hàm để xử lý dữ liệu
learnset = clean_and_prepare_dataset(file_path)

# Hiển thị 10 dòng đầu để kiểm tra
print("\n 10 dòng dữ liệu đầu tiên sau khi xử lý:")
for data in learnset[:10]:
    print(data)
