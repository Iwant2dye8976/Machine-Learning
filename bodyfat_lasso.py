import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

# Load data
df = pd.read_csv(".\\data_sets\\bodyfat.csv")
df = df.dropna()

# Define input features and target variable
X = df[['Density', 'Age', 'Chest', 'Abdomen']]
y = df['BodyFat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X__trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=29)

# Create and train the model
model = Lasso(alpha=.1)
model.fit(X__trained_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
Y = np.array(y_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_ = cross_val_score(model, X, y,cv=5).mean()
print("MSE = %f" % mse)
print("R2 = %f" % r2)
print("CV = %f" % cv_)

# Create a DataFrame for the actual, predicted, and difference values
result_df = pd.DataFrame({
    'Actual': Y,
    'Predicted': y_pred,
    'Difference': abs(Y - y_pred),
    'MSE': mse,
    'R2': r2,
    'Cross-Validation Score': cv_
})

# Export the DataFrame to an Excel file
result_df.to_excel(".\\results\\BodyFat_Lasso.xlsx", index=False)

#  Print the differences (optional)
for i in range(0, len(Y)):
    print('{:.3f} \t\t {:.3f} \t\t {:.3f}'.format(Y[i], y_pred[i], abs(Y[i] - y_pred[i])))

# Plot the comparison between actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual BodyFat', marker='o', color='b')
plt.plot(y_pred, label='Predicted BodyFat', linestyle='--', marker='x', color='r')
plt.title("So sánh giữa giá trị thực tế và dự đoán")
plt.xlabel("Số lượng mẫu")
plt.ylabel("BodyFat")
plt.legend()
plt.grid()
plt.show()