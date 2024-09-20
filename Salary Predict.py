import streamlit as st
import pandas as pd
import pickle

# Tiêu đề cho ứng dụng web
st.title('DỰ ĐOÁN LƯƠNG')

# Form nhập thông số
st.write("Nhập thông số:")

# Tạo các trường nhập liệu cho các thông số (ví dụ với 4 thông số)
age = st.number_input('Tuổi', min_value=20, max_value=65, value=20, step=1)  # Nhập tuổi
gender = st.radio('Giới tính', ('Nam', 'Nữ'))  # Chọn giới tính
education_level = st.selectbox('Chọn trình độ học vấn:', 
                                ("Cử nhân(Bachelor's)", "Thạc sĩ(Master's)", "Tiến sĩ(PhD)")) # Chọn trình độ học vấn
years_of_experience = st.number_input('Số năm kinh nghiệm', min_value=0.0, max_value=age-14, value=0.0, step=1.0)  # Nhập số năm kinh nghiệm
model_type = st.selectbox('Chọn mô hình dự đoán:',
                    ("Linear Regression", "Lasso", "Neuron Network", "Stacking")) # Chọn mô hình dự đoán

# Chuyển đổi giới tính và trình độ học vấn thành số
gender_male = 1 if gender == "Nam" else 0
master = 1 if education_level == "Thạc sĩ(Master's)" else 0
phd = 1 if education_level == "Tiến sĩ(PhD)" else 0
train_model = "Linear Regression" if model_type=="Linear Regression" else "Lasso" if model_type=="Lasso" else "Neuron Network" if model_type=="Neuron Network" else "Stacking"
# Hàm tải mô hình
def load_model(type):
    with open(f'.\\trained models\\{type}_model.pkl', 'rb') as file:
        model = pickle.load(file)  # Tải mô hình từ file
    return model

# Tải mô hình
model = load_model(train_model)

# Nút dự đoán
if st.button('Dự đoán ngay🫵🫵'):
    # Chuẩn bị dữ liệu đầu vào
    input_data = pd.DataFrame([[age, years_of_experience, master, phd, gender_male]], 
                              columns=['Age', 'Years of Experience', "Education Level_Master's", "Education Level_PhD", 'Gender_Male'])
    
    # Thực hiện dự đoán
    prediction = model.predict(input_data)

    # Hiển thị kết quả dự đoán
    st.success(f'Mức lương dự đoán: {round(prediction[0], 2)}$')
