import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sc = preprocessing.MinMaxScaler()

data = df.iloc[:, :-1]
target = df.iloc[:, -1]
data = sc.fit_transform(data)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=2023)

from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(train_x, train_y)

print(est_gp.score(valid_x, valid_y))  # 0.5967867578111098
print(
    est_gp._program)  # sub(div(X5, 0.022), mul(add(mul(X10, X4), sub(X5, 0.502)), sub(div(X10, X12), add(div(mul(add(X5, X5), mul(X12, X10)), sub(0.479, 0.502)), sub(div(X11, X12), sub(0.479, 0.502))))))