import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Tải và chuẩn bị dữ liệu
df = pd.read_csv(".\\data_sets\\bodyfat.csv")
df = df.dropna()
X = df[['Density', 'Age', 'Chest', 'Abdomen']]
y = df['BodyFat']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kf = KFold(n_splits=5, shuffle=True, random_state=29)

# Định nghĩa các mô hình cơ sở
base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=1000, random_state=42))
]

# Định nghĩa meta-learner
meta_learner = Lasso()

# Tạo và huấn luyện mô hình stacking
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)
stacking_model.fit(X_train_scaled, y_train)

# Dự đoán và đánh giá mô hình
y_pred = stacking_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
cv = cross_val_score(stacking_model, X, y, cv=5).mean()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_ = cross_val_score(stacking_model, X, y, cv=5).mean()
print("MSE = %f" % mse)
print("R2 = %f" % r2)
print("CV = %f" % cv_)

# Tạo DataFrame để lưu trữ giá trị thực tế, dự đoán và độ chênh lệch
result_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Difference': abs(y_test - y_pred),
    'MSE': mse,
    'R2': r2,
    'Cross-Validation Score': cv_
})

# Xuất DataFrame ra tệp Excel
result_df.to_excel(".\\results\\BodyFat_StackingModel.xlsx", index=False)

# In ra độ chênh lệch giữa giá trị thực tế và dự đoán (tùy chọn)
for i in range(len(y_test)):
    print('{:.3f} \t\t {:.3f} \t\t {:.3f}'.format(y_test.values[i], y_pred[i], abs(y_test.values[i] - y_pred[i])))

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
