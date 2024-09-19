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
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt
import pickle

evaluation = pd.DataFrame({'Model': [],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'R-squared (test)':[],
                           '5-Fold Cross Validation':[]})

df = pd.read_csv(".\\data_sets\\Salary Data.csv")
df=df.dropna()
df.drop_duplicates(inplace=True)
df=df.drop(['Job Title'],axis=1)
df = pd.get_dummies(df, columns=["Education Level"], drop_first=True)*1

X = df.drop(columns=[ "Salary", "Gender"])
y = df["Salary"]


base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=5000))
]

meta_learner = LinearRegression()

LR = LinearRegression()
LS = Lasso(alpha=1)
MLP = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=5000)
STK = StackingRegressor(estimators=base_models, final_estimator=meta_learner)


def get_Score(model, X_train, X_test, y_train, y_test, kf, model_name, eva):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    cv = cross_val_score(model,X,y,cv=kf).mean()
    eva.loc[len(evaluation)] = [model_name, rmse, r2_train, r2_test, cv]
    eva = evaluation.sort_values(by='5-Fold Cross Validation', ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

trains = {
    "Linear Regression": LR,
    "Lasso": LS,
    "Neuron Network": MLP,
    "Stacking": STK
}


for model_name, model in trains.items():
    get_Score(model, X_train, X_test, y_train, y_test, kf, model_name, evaluation)


print(evaluation)

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line of perfect prediction
# plt.xlabel('Actual Median House Value')
# plt.ylabel('Predicted Median House Value')
# plt.title('Linear Regression Predictions vs Actual')
# plt.grid()
# plt.legend()
# plt.show()

# with open('linear_regression_model.pkl', 'wb') as f:
#     pickle.dump(linear_model, f)

# with open('lasso_model.pkl', 'wb') as f:
#     pickle.dump(lasso_model, f)

# with open('neural_network_model.pkl', 'wb') as f:
#     pickle.dump(nn_model, f)