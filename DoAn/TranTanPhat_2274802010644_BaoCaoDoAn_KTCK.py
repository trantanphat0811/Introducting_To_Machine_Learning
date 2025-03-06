import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import unicodedata
import re

# Đọc dữ liệu
def doc_du_lieu_du_phong(duong_dan):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'cp1258']
    for enc in encodings:
        try:
            data = pd.read_csv(duong_dan, sep='\t', encoding=enc)
            print(f"Đọc thành công với encoding: {enc}")
            return data
        except Exception as e:
            print(f"Lỗi với encoding {enc}: {e}")
    raise Exception("Không thể đọc dữ liệu với các encoding đã thử!")

du_lieu = doc_du_lieu_du_phong('/Users/trantanphat/Documents/Python/NMHM/DoAn/SinhVien.csv')

print(f"Số dòng dữ liệu ban đầu: {len(du_lieu)}")
print("\nThông tin dữ liệu:")
print(du_lieu.info())
print("\n5 dòng đầu tiên:")
print(du_lieu.head())

# Làm sạch dữ liệu
def lam_sach_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'_', ' ', text)
    return text

cac_cot_text = ['Gioi_tinh', 'Chuyen_nganh', 'Ton_giao', 'Noi_song', 'Quoc_tich']
for cot in cac_cot_text:
    du_lieu[cot] = du_lieu[cot].apply(lam_sach_text)

du_lieu.drop_duplicates(inplace=True)
du_lieu.dropna(inplace=True)
du_lieu = du_lieu[du_lieu['Noi_song'] != 'Khac']
du_lieu = du_lieu[
    (du_lieu['GPA'].between(0, 4.0)) &
    (du_lieu['Tuoi'].between(17, 30)) &
    (du_lieu['So_tin_chi_truot'] >= 0) &
    (du_lieu['So_buoi_nghi'] >= 0)
]

# Thêm cột xếp loại học lực và điều kiện tốt nghiệp
du_lieu['Hoc_luc'] = du_lieu['GPA'].apply(lambda gpa: (
    'Xuat sac' if gpa >= 3.6 else
    'Gioi' if gpa >= 3.2 else
    'Kha' if gpa >= 2.5 else
    'Trung binh' if gpa >= 2.0 else
    'Yeu'
))

du_lieu['Dieu_kien_tot_nghiep'] = du_lieu.apply(
    lambda row: 'Dat' if row['GPA'] >= 2.0 and row['So_tin_chi_truot'] <= 10 and row['So_buoi_nghi'] <= 15 else 'Khong Dat',
    axis=1
)

# Mã hóa dữ liệu
bang_ma_hoa = {
    'Gioi_tinh': {'Nam': 0, 'Nu': 1},
    'Chuyen_nganh': {'Cong nghe thong tin': 0, 'Ke toan': 1, 'Quan tri kinh doanh': 2},
    'Ton_giao': {'Khong': 0, 'Phat giao': 1, 'Thien chua giao': 2},
    'Noi_song': {'TP.HCM': 0, 'Ha Noi': 1, 'Da Nang': 2},
    'Quoc_tich': {'Viet Nam': 0, 'Lao': 1, 'Campuchia': 2},
    'Hoc_luc': {'Xuat sac': 0, 'Gioi': 1, 'Kha': 2, 'Trung binh': 3, 'Yeu': 4},
    'Dieu_kien_tot_nghiep': {'Dat': 1, 'Khong Dat': 0}
}
for cot, ma_hoa in bang_ma_hoa.items():
    du_lieu[cot] = du_lieu[cot].map(ma_hoa)

du_lieu.dropna(inplace=True)

# Thêm cột Nguy cơ rớt tốt nghiệp
du_lieu['Nguy_co_rot_tot_nghiep'] = du_lieu.apply(
    lambda row: 1 if row['GPA'] < 2.0 or row['So_buoi_nghi'] > 15 else 0,
    axis=1
)

if du_lieu.empty:
    raise Exception("Dữ liệu trống sau khi xử lý! Kiểm tra lại dữ liệu đầu vào và các bước làm sạch.")

# Chuẩn bị dữ liệu huấn luyện
dac_trung = ["Tuoi", "Gioi_tinh", "GPA", "So_tin_chi_truot",
             "So_buoi_nghi", "Chuyen_nganh", "Ton_giao", "Nien_khoa", 
             "Noi_song", "Quoc_tich", "Hoc_luc", "Dieu_kien_tot_nghiep"]

X = du_lieu[dac_trung]
y = du_lieu["Nguy_co_rot_tot_nghiep"]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
mo_hinh = DecisionTreeClassifier(max_depth=4, random_state=42)
mo_hinh.fit(X_train, y_train)

# Đánh giá mô hình
y_du_doan = mo_hinh.predict(X_test)
print(f"\n Độ chính xác: {accuracy_score(y_test, y_du_doan):.2f}")
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_du_doan))

# Vẽ cây quyết định
plt.figure(figsize=(30, 20))
plot_tree(
    mo_hinh, 
    feature_names=dac_trung,
    class_names=["Khong rot tot nghiep", "Nguy co rot tot nghiep"],
    filled=True, 
    rounded=True,
    fontsize=14
)
plt.title("Cay quyet dinh nguy co rot tot nghiep cua sinh vien", fontsize=22)
plt.show()

# In kết quả dự đoán kèm tên sinh viên
ket_qua = X_test.copy()
ket_qua['Ten_sinh_vien'] = du_lieu.loc[X_test.index, 'Ten_sinh_vien']
ket_qua['Thuc_te'] = y_test.values
ket_qua['Du_doan'] = y_du_doan

print("\n Kết quả dự đoán:")
print(ket_qua[['Ten_sinh_vien', 'Thuc_te', 'Du_doan']])
