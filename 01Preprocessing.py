import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('bolt.csv')
df = df.iloc[:, [0,1, 2, 3, 4,5,8]]
X = df.iloc[:, [0, 2, 3, 4]].values
y = df.iloc[:, -1].values
m = X.shape[1]
n = X.shape[0]
plt.figure(figsize=(8, 6))
sns.set(font='Times New Roman')
ax = sns.heatmap(df.corr(), annot=True, cmap='Greys', xticklabels=df.corr().columns.str.replace('A', 'Subscript[A]').str.replace('B', 'Subscript[B]'), yticklabels=df.corr().columns.str.replace('A', 'Subscript[A]').str.replace('B', 'Subscript[B]'), cbar=True, annot_kws={'size': 16})
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=20)
ax.set_xticklabels(ax.get_xticklabels(), size=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.show()
df = pd.read_csv('bolt.csv')
X = df.iloc[:, [0, 1, 2, 3, 4, 5,8]].values
Num = df.iloc[:, 6].values
y = df.iloc[:, -1].values
cov_matrix = np.cov(X.T)
m_dist = []
for sample in X:
    dist = sps.distance.mahalanobis(sample, np.mean(X, axis=0), cov_matrix)
    m_dist.append(dist)
data = pd.DataFrame({'Num': Num, 'Mahalanobis Distance': np.array(m_dist)})
data.to_excel('result/mashi.xlsx', index=False)
plt.scatter(Num, np.array(m_dist) / 1e4)
plt.xlabel('Samples')
plt.ylabel('Mahalanobis Distance (x10^4)')
plt.show()
max_dist_indices = sorted(range(len(m_dist)), key=lambda i: m_dist[i], reverse=True)[:20]
max_distances = [m_dist[i] for i in max_dist_indices]
corresponding_num = [Num[i] for i in max_dist_indices]
for dist, num in zip(max_distances, corresponding_num):
    print(dist,num)

