import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
df = pd.read_csv(".\\data_sets\\Salary Data.csv")
df=df.dropna()
df.drop_duplicates(inplace=True)
df=df.drop(['Job Title'],axis=1)
df = pd.get_dummies(df, columns=["Education Level"], drop_first=True)*1

X = df.drop(columns=[ "Salary", "Gender"])
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_pred = model.predict(X_test)
mselin = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)
cv = cross_val_score(model,X,y,cv=kf).mean()
print(f'Linear Regression - MSE: {mselin}, R²: {lin_r2}, CV-Score: {cv}')

model = Lasso(alpha=1)
model.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_pred = model.predict(X_test)
mselin = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)
cv = cross_val_score(model,X,y,cv=kf).mean()
print(f'Lasso - MSE: {mselin}, R²: {lin_r2}, CV-Score: {cv}')

model = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mselin = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)
cv = cross_val_score(model,X,y,cv=kf).mean()
print(f'Neuron Network - MSE: {mselin}, R²: {lin_r2}, CV-Score: {cv}')

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line of perfect prediction
# plt.xlabel('Actual Median House Value')
# plt.ylabel('Predicted Median House Value')
# plt.title('Linear Regression Predictions vs Actual')
# plt.grid()
# plt.legend()
# plt.show()