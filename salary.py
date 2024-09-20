import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Bỏ qua cảnh báo
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
import pickle
import os

# Khởi tạo DataFrame để lưu trữ kết quả đánh giá
evaluation = pd.DataFrame({
    'Model': [],
    'Root Mean Squared Error (RMSE)': [],
    'R-squared (training)': [],
    'R-squared (test)': [],
    '5-Fold Cross Validation': []
})

# Tải dữ liệu
df = pd.read_csv(".\\data_sets\\Salary Data.csv")

# Xử lý dữ liệu
df = df.dropna()  # Bỏ các dòng có giá trị NaN
df.drop_duplicates(inplace=True)  # Bỏ các dòng trùng lặp
df = df.drop(['Job Title'], axis=1)  # Bỏ cột 'Job Title'

# Xử lý dữ liệu cho cột 'Education Level' và 'Gender'
df = pd.get_dummies(df, columns=["Education Level", "Gender"], drop_first=True) * 1

# Định nghĩa biến đầu vào và biến mục tiêu
X = df.drop(columns=["Salary"])  # Biến đầu vào
y = df["Salary"]  # Biến mục tiêu

# Định nghĩa các mô hình cơ bản
base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=5000))
]

# Định nghĩa mô hình meta-learner
meta_learner = LinearRegression()

# Khởi tạo các mô hình
LR = LinearRegression()
LS = Lasso(alpha=10)
MLP = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=5000)
STK = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

# Hàm huấn luyện mô hình
def Model_Train(model, X_train, X_test, y_train, y_test, kf, model_name, eva):
    model.fit(X_train, y_train)  # Huấn luyện mô hình
    y_pred_train = model.predict(X_train)  # Dự đoán trên tập huấn luyện
    y_pred_test = model.predict(X_test)  # Dự đoán trên tập kiểm tra
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # Tính RMSE
    r2_train = r2_score(y_train, y_pred_train)  # Tính R² trên tập huấn luyện
    r2_test = r2_score(y_test, y_pred_test)  # Tính R² trên tập kiểm tra
    cv = cross_val_score(model, X, y, cv=kf).mean()  # Tính điểm Cross Validation
    eva.loc[len(evaluation)] = [model_name, rmse, r2_train, r2_test, cv]  # Lưu kết quả vào DataFrame

    # Lưu mô hình nếu điều kiện thỏa mãn
    if (abs(r2_test - cv) < .2 and r2_test > .7):
        save_path = os.path.join(".\\trained models", f'{model_name}_model.pkl')  # Đường dẫn lưu mô hình
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)  # Lưu mô hình vào file

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Định nghĩa KFold

# Khởi tạo từ điển chứa các mô hình
trains = {
    "Linear Regression": LR,
    "Lasso": LS,
    "Neuron Network": MLP,
    "Stacking": STK
}

# Huấn luyện từng mô hình
for model_name, model in trains.items():
    Model_Train(model, X_train, X_test, y_train, y_test, kf, model_name, evaluation)

# In kết quả đánh giá mô hình sắp xếp giảm dần theo 5-Fold CV
print(evaluation.sort_values(by='5-Fold Cross Validation', ascending=False))
