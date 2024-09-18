import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

# Đọc dữ liệu
df = pd.read_csv(".\\data_sets\\bodyfat.csv")
df = df.dropna()

# Xác định các đặc trưng đầu vào và biến mục tiêu
X = df[['Density', 'Age', 'Chest', 'Abdomen']]
y = df['BodyFat']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X__trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=29)

# Tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X__trained_scaled, y_train)

# Dự đoán giá trị
y_pred = model.predict(X_test_scaled)
Y = np.array(y_test)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_ = cross_val_score(model, X, y, cv=5).mean()
print("MSE = %f" % mse)
print("R2 = %f" % r2)
print("CV = %f" % cv_)

# Tạo DataFrame để lưu trữ giá trị thực tế, dự đoán, và độ chênh lệch
result_df = pd.DataFrame({
    'Actual': Y,
    'Predicted': y_pred,
    'Difference': abs(Y - y_pred),
    'MSE': mse,
    'R2': r2,
    'Cross-Validation Score': cv_
})

# Xuất DataFrame ra tệp Excel
# result_df.to_excel(".\\results\\BodyFat_Linear_Regression.xlsx", index=False)

# In ra độ chênh lệch giữa giá trị thực tế và dự đoán (tùy chọn)
for i in range(0, len(Y)):
    print('{:.3f} \t\t {:.3f} \t\t {:.3f}'.format(Y[i], y_pred[i], abs(Y[i] - y_pred[i])))

# Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Giá trị thực tế BodyFat', marker='o', color='b')
plt.plot(y_pred, label='Dự đoán BodyFat', linestyle='--', marker='x', color='r')
plt.title("So sánh giữa giá trị thực tế và dự đoán")
plt.xlabel("Số lượng mẫu")
plt.ylabel("BodyFat")
plt.legend()
plt.grid()
plt.show()

# Uncomment để vẽ Learning Curve nếu cần
# train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)

# train_mean = np.mean(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, label='Điểm huấn luyện', color='b')
# plt.plot(train_sizes, test_mean, label='Điểm kiểm tra chéo', color='r')

# plt.title('Learning Curve')
# plt.xlabel('Kích thước tập huấn luyện')
# plt.ylabel('Điểm')
# plt.legend(loc='best')
# plt.grid()
# plt.show()
