import pandas as pd

# Tạo dữ liệu mẫu phù hợp với mô hình
du_lieu_mau = pd.DataFrame({
    'Tuoi': [20, 22, 21, 23, 19],
    'Gioi_tinh': [1, 0, 1, 0, 1],
    'GPA': [2.8, 1.9, 3.5, 1.5, 3.0],
    'So_tin_chi_truot': [5, 12, 3, 15, 2],
    'So_buoi_nghi': [8, 16, 5, 20, 4],
    'Chuyen_nganh': [0, 1, 2, 0, 1],
    'Ton_giao': [0, 1, 0, 2, 0],
    'Nien_khoa': [2021, 2020, 2021, 2019, 2022],
    'Noi_song': [0, 1, 2, 0, 1],
    'Quoc_tich': [0, 0, 0, 0, 0],
    'Hoc_luc': [2, 4, 1, 4, 2],
    'Ten_sinh_vien': ['Nguyen Van A', 'Tran Thi B', 'Le Van C', 'Pham Thi D', 'Hoang Van E'],
    'Thuc_te': [0, 1, 0, 1, 0],
    'Du_doan': [0, 1, 0, 1, 0]
})

# Xuất file CSV
du_lieu_mau.to_csv('/Users/trantanphat/Documents/Python/NMHM/DoAn/ket_qua_du_doan.csv', index=False)


