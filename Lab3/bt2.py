def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a): #Tính xác suất có điều kiện P(A|B) theo định lý Bayes
    if not (0 <= p_a <= 1 and 0 <= p_b_given_a <= 1 and 0 <= p_b_given_not_a <= 1): #Tính xác suất có điều kiện hợp lệ
        raise ValueError("All probabilities must be between 0 and 1")
    
    not_a = 1 - p_a #Tính xác suất của A
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a #Tính xác suất tổng hợp 
    
    if p_b == 0: #Kiểm tra lỗi
        raise ValueError("p(B) cannot be zero (division by zero error)")
    
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

# Thông số đầu vào
p_a = 0.0002
p_b_given_a = 0.85
p_b_given_not_a = 0.05

# Tính toán
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)

# Hiển thị kết quả
print(f"P(A|B) = {result * 100:.3f}%")
#Dù xét nghiệm có độ nhạy cao, khi các căn bệnh hiếm gặp thì phần lớn kết quả dương tính sai
# Định lý bayes giúp hiểu rằng tỉ lệ dương tính giả có thể gây hiểu lầm về dịch bệnh 