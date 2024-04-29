import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from deap import creator, base, tools, algorithms
data = pd.read_csv('bolt.csv')
m = data.shape[1] - 1 
n = data.shape[0]
# For example, referring to the line below, change the number of parameters in this line.
X = data.iloc[:, [0, 2]]
y = data.iloc[:, m] 
def evaluate(individual):
    n_estimators = individual[0]
    max_depth = individual[1]
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=40
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse,
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("n_estimators", np.random.randint, low=1, high=300)
toolbox.register("max_depth", np.random.randint, low=1, high=10)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=[100, 2], up=[300, 10], indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
pop_size = 500
n_generations = 20
population = toolbox.population(n=pop_size)
for gen in range(n_generations):
    print("Generation:", gen + 1)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
best_individual = tools.selBest(population, k=1)[0]
best_n_estimators = best_individual[0]
best_max_depth = best_individual[1]
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.1,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=40
)
model.fit(X_train, y_train)
model.save_model('xgboost_model.model')
preds = model.predict(X_test)
plt.scatter(y_train, model.predict(X_train), c='blue', label='Train') 
plt.scatter(y_test, preds, c='red', label='Test') 
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--') 
plt.xlabel('Target Pu(kN)') 
plt.ylabel('Predicted Pu(kN)') 
plt.legend() 
plt.show()
