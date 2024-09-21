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
import pickle
import openpyxl

# Khởi tạo DataFrame để lưu trữ kết quả đánh giá
evaluation = pd.DataFrame({
    'Model': [],
    'Root Mean Squared Error (RMSE)': [],
    'Mean Absolute Error (MAE)': [],
    'R-squared (training)': [],
    'R-squared (test)': [],
    'Nash-Sutcliffe Efficiency(NSE)': []
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

# Định nghĩa cho Stacking
base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000))
]
# Định nghĩa mô hình meta-learner
meta_learner = Lasso(alpha=1)

LR = LinearRegression()
LS = Lasso(alpha=1)
MLP = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
STK = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

# Khởi tạo từ điển chứa các mô hình
trains = {
    "Linear Regression": LR,
    "Lasso": LS,
    "Neuron Network": MLP,
    "Stacking": STK
}


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def nash_sutcliffe_efficiency(y_true, y_pred):
    #Hàm tính Nash-Sutcliffe Efficiency (NSE)
    # Tính số dư bình phương của dự đoán
    numerator = np.sum((y_true - y_pred) ** 2)
    
    # Tính số dư bình phương của trung bình giá trị thực tế
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Tính NSE
    nse = 1 - numerator / denominator
    
    return nse

# Hàm huấn luyện mô hình
def Model_Train(model, X_train, X_test, y_train, y_test, model_name):
    global evaluation  # Sử dụng biến global để cập nhật trực tiếp DataFrame
    model.fit(X_train, y_train)  # Huấn luyện mô hình
    y_pred_train = model.predict(X_train)  # Dự đoán trên tập huấn luyện
    y_pred_test = model.predict(X_test)  # Dự đoán trên tập kiểm tra
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # Tính RMSE
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)  # Tính R² trên tập huấn luyện
    r2_test = r2_score(y_test, y_pred_test)  # Tính R² trên tập kiểm tra
    nse = nash_sutcliffe_efficiency(y_test, y_pred_test)
    evaluation.loc[len(evaluation)] = [model_name, rmse, mae, r2_train, r2_test, nse]  # Lưu kết quả vào DataFrame

    # Lưu mô hình nếu điều kiện thỏa mãn
    if (r2_test > .7):
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(".\\trained models", exist_ok=True)
        save_path = os.path.join(".\\trained models", f'{model_name}_model.pkl')  # Đường dẫn lưu mô hình
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)  # Lưu mô hình vào file

if __name__ == "__main__":
    # Huấn luyện từng mô hình
    for model_name, model in trains.items():
        Model_Train(model, X_train, X_test, y_train, y_test, model_name)


    evaluation_sorted = evaluation.sort_values(by=['Root Mean Squared Error (RMSE)','Mean Absolute Error (MAE)','R-squared (test)'], ascending=[True,True,False])
    # In kết quả đánh giá mô hình sắp xếp giảm dần theo 5-Fold CV
    print(evaluation_sorted)

    #Lưu kết quả vào folder results
    evaluation_sorted.to_excel(".\\results\\KQ_danh_gia.xlsx", index=False)

