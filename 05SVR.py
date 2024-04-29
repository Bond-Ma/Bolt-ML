import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
df = pd.read_csv('bolt.csv')
X = df.iloc[:, [0, 2, 3, 4]].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=94)
def svr_obj(params):
    svr = SVR(kernel='rbf', degree=int(params[0]), C=params[1], epsilon=params[2])
    svr.fit(X_train, y_train)
    y_pred_train = svr.predict(X_train)
    return np.sqrt(mean_squared_error(y_train, y_pred_train))
bounds = [(1, 5), (1, 500), (0.001, 1)]
result = differential_evolution(svr_obj, bounds)
best_degree = int(result.x[0])
best_C = result.x[1]
best_epsilon = result.x[2]
svr_best = SVR(kernel='rbf', degree=best_degree, C=best_C, epsilon=best_epsilon)
svr_best.fit(X_train, y_train)
y_pred_train = svr_best.predict(X_train)
y_pred_test = svr_best.predict(X_test)
svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.01, C=180.0, epsilon=0.7, shrinking=True,
        cache_size=10000, verbose=False, max_iter=-1)
svr.fit(X_train, y_train)
import pickle
pickle.dump(svr, open('svr_model.pkl', 'wb'))
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)
plt.scatter(y_train, y_pred_train, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
plt.xlabel('Target Pu(kN)')
plt.ylabel('Predicted Pu(kN)')
plt.legend()
plt.show()