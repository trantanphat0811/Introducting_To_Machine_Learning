import math

# Log cơ số 2 (bit)
p = 0.5
h = -math.log2(p)
print('Log2:  p(x)=%.3f, information: %.3f bits' % (p, h))

# Log cơ số 10 (hartley)
p = 0.5
h = -math.log10(p)
print('Log10: p(x)=%.3f, information: %.3f hartleys' % (p, h))

# Log tự nhiên (nat)
p = 0.5
h = -math.log(p)  # log tự nhiên (ln)
print('Ln:    p(x)=%.3f, information: %.3f nats' % (p, h))

#Đơn vị bits phổ biến nhất trong lý thuyết thông tin, học máy, và xử lý dữ liệu
#Nats, và Hartleys cũng được sử dụng trong toán học hoặc thông tin lượng tử    
