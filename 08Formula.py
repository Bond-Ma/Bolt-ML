import numpy as np
import pandas as pd
import random
data = pd.read_csv('bolt19.csv')
d = data['d'].values
fu = data['fu'].values
h = data['h'].values
Pu = data['Pu'].values
def fitness(a, b):
    p = Pu - d*fu*(a*d + b*h)/1000
    return np.sum(p**2)
pop_size = 5000
chrom_length = 2
pc = 0.8
pm = 0.2
max_iter = 20
populations = []
for i in range(pop_size):
    chromosome = []
    for j in range(chrom_length):
        chromosome.append(random.uniform(0.0, 1.0))
    populations.append(chromosome)
for iter in range(max_iter):
    fits = []
    for chromosome in populations:
        a, b = chromosome
        fits.append(fitness(a, b))
    fits, populations = zip(*sorted(zip(fits, populations)))
    best_fit = fits[0]
    best_chromosome = populations[0]
    new_populations = []
    for i in range(pop_size//2):
        parent1 = random.choice(populations[:pop_size//2])
        parent2 = random.choice(populations[:pop_size//2])
        if random.random() < pc:
            pos = random.randint(0, chrom_length-1)
            child1 = parent1[0:pos] + parent2[pos:]
            child2 = parent2[0:pos] + parent1[pos:]
        else:
            child1 = parent1
            child2 = parent2
        if random.random() < pm:
            pos = random.randint(0, chrom_length-1)
            child1[pos] = random.uniform(0.0, 1.0)
            child2[pos] = random.uniform(0.0, 1.0)
        new_populations.append(child1)
        new_populations.append(child2)
    populations = new_populations
a, b = best_chromosome
print("最优解：a = %.4f, b = %.4f，适应度：%.4f" % (a, b, best_fit))
data = pd.read_csv('bolt20.csv')
d = data['d'].values
fu = data['fu'].values
h = data['h'].values
Pu = data['Pu'].values
def fitness(a, b):
    p = Pu - d*fu*(a*d + b*h)/1000
    return np.sum(p**2)
pop_size = 500
chrom_length = 2
pc = 0.8
pm = 0.2
max_iter = 20
populations = []
for i in range(pop_size):
    chromosome = []
    for j in range(chrom_length):
        chromosome.append(random.uniform(0.0, 1.0))
    populations.append(chromosome)
for iter in range(max_iter):
    fits = []
    for chromosome in populations:
        a, b = chromosome
        fits.append(fitness(a, b))
    fits, populations = zip(*sorted(zip(fits, populations)))
    best_fit = fits[0]
    best_chromosome = populations[0]
    new_populations = []
    for i in range(pop_size//2):
        parent1 = random.choice(populations[:pop_size//2])
        parent2 = random.choice(populations[:pop_size//2])
        if random.random() < pc:
            pos = random.randint(0, chrom_length-1)
            child1 = parent1[0:pos] + parent2[pos:]
            child2 = parent2[0:pos] + parent1[pos:]
        else:
            child1 = parent1
            child2 = parent2
        if random.random() < pm:
            pos = random.randint(0, chrom_length-1)
            child1[pos] = random.uniform(0.0, 1.0)
            child2[pos] = random.uniform(0.0, 1.0)
        new_populations.append(child1)
        new_populations.append(child2)
    populations = new_populations
a, b = best_chromosome
print("最优解：a = %.4f, b = %.4f，适应度：%.4f" % (a, b, best_fit))