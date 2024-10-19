import numpy as np
import pandas as pd
import warnings
import os

# Bỏ qua cảnh báo
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error ,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt
import pickle
import openpyxl

# Tải dữ liệu
df = pd.read_csv(".\\data_sets\\Salary Data.csv")

# Xử lý dữ liệu
df = df.dropna()  # Bỏ các dòng có giá trị NaN
df.drop_duplicates(inplace=True)  # Bỏ các dòng trùng lặp

# Loại bỏ cột 'Job Title'
df = df.drop(['Job Title'],axis=1)

# Xử lý dữ liệu cho cột 'Education Level' và 'Gender'
df = pd.get_dummies(df, columns=["Education Level", "Gender"], drop_first=True) * 1

# Định nghĩa biến đầu vào và biến mục tiêu
X = df.drop(columns=["Salary"])  # Biến đầu vào
y = df["Salary"]  # Biến mục tiêu

# Định nghĩa các mô hình cơ bản

# Định nghĩa cho Stacking
base_learners = [
    ('lr', LinearRegression()),
    ('lasso', Lasso(alpha=10)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), activation="relu", alpha=0.001, learning_rate="constant", solver="lbfgs" ,max_iter=1000 ,random_state=42))
]
# Định nghĩa mô hình meta-learner
meta_model = Lasso(alpha=10)
# meta_model = LinearRegression()


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
score = np.mean(cross_val_score(model, X, y, cv=kfold))

print(f'CrossValidation Score: %f'% score )
print(f'R2 Score(Train): %f'% r2_train)
print(f'R2 Score(Test): %f'% r2_test)
print(f'RMSE(Test): %f'% rmse)
# CrossValidation Score: 0.890352
# R2 Score(Train): 0.899968
# R2 Score(Test): 0.894333
# RMSE(Test): 14131.082950

os.makedirs("D:\\SalaryPredict", exist_ok=True)
save_path = os.path.join("D:\\SalaryPredict", 'stacking_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model, f) 

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

# Tính giá trị trung bình và độ lệch chuẩn cho tập huấn luyện và tập kiểm thử
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Chuyển đổi từ MSE sang RMSE (lấy căn bậc 2 và đổi dấu)
train_scores_rmse = np.sqrt(-train_scores_mean)
test_scores_rmse = np.sqrt(-test_scores_mean)

# Vẽ biểu đồ
plt.figure()
plt.fill_between(train_sizes, train_scores_rmse - np.sqrt(train_scores_std),
                 train_scores_rmse + np.sqrt(train_scores_std), alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_rmse - np.sqrt(test_scores_std),
                 test_scores_rmse + np.sqrt(test_scores_std), alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_rmse, 'o-', color="r", label="Training RMSE")
plt.plot(train_sizes, test_scores_rmse, 'o-', color="g", label="Cross-validation RMSE")

# Thêm các nhãn và tiêu đề
plt.title("Learning Curve with RMSE for Linear Regression")
plt.xlabel("Training examples")
plt.ylabel("RMSE")
plt.legend(loc="best")
plt.grid()

# Hiển thị biểu đồ
plt.show()