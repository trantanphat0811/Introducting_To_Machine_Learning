import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import unicodedata
import re
import os

# Đọc dữ liệu với các encoding dự phòng
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

# Đường dẫn file dữ liệu
duong_dan_file = '/Users/trantanphat/Documents/Python/NMHM/DoAn/SinhVien.csv'
du_lieu = doc_du_lieu_du_phong(duong_dan_file)

# Làm sạch dữ liệu văn bản
def lam_sach_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'_', ' ', text)
    return text

cot_text = ['Gioi_tinh', 'Chuyen_nganh', 'Ton_giao', 'Noi_song', 'Quoc_tich']
for cot in cot_text:
    du_lieu[cot] = du_lieu[cot].apply(lam_sach_text)

# Xử lý dữ liệu thiếu và giá trị không hợp lệ
du_lieu.drop_duplicates(inplace=True)
du_lieu.dropna(inplace=True)
du_lieu = du_lieu[du_lieu['Noi_song'] != 'Khac']
du_lieu = du_lieu[
    (du_lieu['GPA'].between(0, 4.0)) &
    (du_lieu['Tuoi'].between(17, 30)) &
    (du_lieu['So_tin_chi_truot'] >= 0) &
    (du_lieu['So_buoi_nghi'] >= 0)
]

# Thêm cột học lực và điều kiện tốt nghiệp
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

# Mã hóa LabelEncoder
label_cols = ['Gioi_tinh', 'Chuyen_nganh', 'Ton_giao', 'Noi_song', 'Quoc_tich', 'Hoc_luc', 'Dieu_kien_tot_nghiep']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    du_lieu[col] = le.fit_transform(du_lieu[col])
    label_encoders[col] = le

# Thêm cột nguy cơ rớt tốt nghiệp
du_lieu['Nguy_co_rot_tot_nghiep'] = du_lieu.apply(
    lambda row: 1 if row['GPA'] < 2.0 or row['So_buoi_nghi'] > 15 else 0,
    axis=1
)

if du_lieu.empty:
    raise Exception("Dữ liệu trống sau khi xử lý!")

# Tách đặc trưng và nhãn
dac_trung = ["Tuoi", "Gioi_tinh", "GPA", "So_tin_chi_truot",
             "So_buoi_nghi", "Chuyen_nganh", "Ton_giao", "Nien_khoa", 
             "Noi_song", "Quoc_tich", "Hoc_luc", "Dieu_kien_tot_nghiep"]

X = du_lieu[dac_trung]
y = du_lieu["Nguy_co_rot_tot_nghiep"]

# Kiểm tra số mẫu lớp thiểu số để chọn k_neighbors phù hợp
n_minority = y.value_counts().min()
k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1

# Áp dụng SMOTE nếu đủ mẫu
if n_minority > 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
else:
    print("Không đủ mẫu để dùng SMOTE, dùng dữ liệu gốc.")
    X_res, y_res = X, y

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# GridSearch tối ưu mô hình
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

mo_hinh = grid_search.best_estimator_
print(f"Thông số tối ưu: {grid_search.best_params_}")

# Đánh giá mô hình
y_du_doan = mo_hinh.predict(X_test)
print(f"\nĐộ chính xác: {accuracy_score(y_test, y_du_doan):.2f}")
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_du_doan))

# Ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_du_doan), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Không rớt tốt nghiệp", "Nguy cơ rớt tốt nghiệp"],
            yticklabels=["Không rớt tốt nghiệp", "Nguy cơ rớt tốt nghiệp"])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Biểu đồ độ quan trọng các đặc trưng
importances = mo_hinh.feature_importances_
features = pd.Series(importances, index=dac_trung).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=features.values, y=features.index, palette='viridis')
plt.title('Biểu đồ độ quan trọng của các đặc trưng', fontsize=16)
plt.xlabel('Mức độ quan trọng')
plt.ylabel('Đặc trưng')
plt.show()

# Biểu đồ ROC Curve
y_proba = mo_hinh.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Biểu đồ ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Biểu đồ Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Biểu đồ Precision-Recall Curve')
plt.grid()
plt.show()

# Vẽ cây quyết định
plt.figure(figsize=(30, 20))
plot_tree(
    mo_hinh, 
    feature_names=dac_trung,
    class_names=["Không rớt tốt nghiệp", "Nguy cơ rớt tốt nghiệp"],
    filled=True, 
    rounded=True,
    fontsize=14
)
plt.title("Cây quyết định nguy cơ rớt tốt nghiệp của sinh viên", fontsize=22)
plt.show()

# In kết quả dự đoán nếu có tên sinh viên
if 'Ten_sinh_vien' in du_lieu.columns:
    ket_qua = X_test.copy()
    ket_qua['Ten_sinh_vien'] = du_lieu.loc[X_test.index, 'Ten_sinh_vien'].values
    ket_qua['Thuc_te'] = y_test.values
    ket_qua['Du_doan'] = y_du_doan
    print("\nKết quả dự đoán:")
    print(ket_qua[['Ten_sinh_vien', 'Thuc_te', 'Du_doan']])
else:
    print("\nCột 'Ten_sinh_vien' không tồn tại trong dữ liệu.")
