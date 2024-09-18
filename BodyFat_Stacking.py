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

# Load and prepare data
df = pd.read_csv(".\\data_sets\\bodyfat.csv")
df = df.dropna()
X = df[['Density', 'Age', 'Chest', 'Abdomen']]
y = df['BodyFat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kf = KFold(n_splits=5, shuffle=True, random_state=29)

# Define base models
base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=1000, random_state=42))
]

# Define meta-learner
meta_learner = Lasso()

# Create and fit the stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)
stacking_model.fit(X_train_scaled, y_train)
Y = np.array(y_test)

# Predict and evaluate the model
y_pred = stacking_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
cv = cross_val_score(stacking_model, X, y,cv=5).mean()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_ = cross_val_score(stacking_model, X, y,cv=5).mean()
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
result_df.to_excel(".\\results\\BodyFat_StackingModel.xlsx", index=False)

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