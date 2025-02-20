import math
import matplotlib.pyplot as plt

# Dữ liệu về số người đến đúng giờ (in_time) và trễ (too_late)
in_time = [(0, 22), (1, 19), (2, 17), (3, 18),
           (4, 16), (5, 15), (6, 9), (7, 7),
           (8, 4), (9, 3), (10, 3), (11, 2)]
too_late = [(6, 6), (7, 9), (8, 12), (9, 17), 
            (10, 18), (11, 15), (12,16), (13, 7),
            (14, 8), (15, 5)]

# Tách dữ liệu để vẽ biểu đồ
X, Y = zip(*in_time)
X2, Y2 = zip(*too_late)

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 5))
plt.bar(X, Y, width=0.9, color="blue", alpha=0.75, label="In Time")
plt.bar(X2, Y2, width=0.8, color="red", alpha=0.75, label="Too Late")
plt.xlabel("Minutes Before Train Leaves")
plt.ylabel("Number of People")
plt.title("Train Arrival Time Analysis")
plt.legend(loc='upper right')
plt.show()

# Chuyển dữ liệu thành dictionary để dễ tra cứu
in_time_dict = dict(in_time)
too_late_dict = dict(too_late)

# Hàm tính xác suất bắt kịp tàu
def catch_the_train(min):
    s = in_time_dict.get(min, 0)  # Số người đến đúng giờ
    m = too_late_dict.get(min, 0)  # Số người đến trễ
    if s == 0:  # Nếu không có ai đến đúng giờ tại phút đó
        return 0
    return s / (s + m)  # Xác suất = số người đến đúng giờ / tổng số người

# Kiểm tra với các giá trị từ -1 đến 15 phút trước khi tàu rời
for minutes in range(-1, 16):
    print(f"{minutes} minutes before train: {catch_the_train(minutes):.2f}")


#0-5m: Xác suất 100% vì không có ai đến trễ
#6m: Xác suất giảm còn 60% vì có người đến trễ
#7-11m: Xác suất ngày càng tăng khi người đến trễ ngày càng nhiều
#12-15m: Xác suất 0% vì có ai đến đúng giờ 

