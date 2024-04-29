import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from deap import algorithms, base, creator, tools
data = pd.read_csv('bolt.csv')
X = data.iloc[:, [0, 2, 3, 4]] 
y = data.iloc[:, -1] 
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def evaluate(individual):
    n_estimators = int(individual[0])
    max_depth = int(individual[1])
    min_samples_split = int(individual[2])
    min_samples_leaf = int(individual[3])
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_features='sqrt',
                               random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = MAPE(y_test, y_pred)
    return (r2, rmse, mae, mape)
POP_SIZE = 50
GENE_SIZE = 4
CXPB = 0.8
MUTPB = 0.2
NGEN = 50
MU = 10
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.randint, 1, 1000)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=GENE_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=1000, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=POP_SIZE-MU, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
best_individual = hof[0]
n_estimators = int(best_individual[0])
max_depth = int(best_individual[1])
min_samples_split = int(best_individual[2])
min_samples_leaf = int(best_individual[3])
print(n_estimators,max_depth,min_samples_split,min_samples_leaf)
rf = RandomForestRegressor(n_estimators=n_estimators,
                           max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           max_features='sqrt',
                           random_state=42)
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
plt.scatter(y_train, y_pred_train, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
plt.xlabel('Target Pu(kN)')
plt.ylabel('Predicted Pu(kN)')
plt.legend()
plt.show()
