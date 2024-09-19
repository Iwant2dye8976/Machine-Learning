import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import StackingRegressor
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

base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=5000))
]

meta_learner = LinearRegression()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)
stacking_model.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = stacking_model.predict(X_test)

mselin = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)
cv = cross_val_score(stacking_model,X,y,cv=kf).mean()
print(f'Linear Regression - MSE: {mselin}, R²: {lin_r2}, CV-Score: {cv}')