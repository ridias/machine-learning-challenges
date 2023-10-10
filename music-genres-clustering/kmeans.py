from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

dataset = pd.read_csv("genres_v2.csv")

print(dataset.info())
print(dataset.isnull().sum())

X = dataset[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Codo de Jambú")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.show()

clustering = KMeans(n_clusters=5, max_iter=300)
Y = clustering.fit_predict(X)
print(Y)
print(X.head())

# Aplica PCA para reducir la dimensionalidad a 2 componentes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Grafica los puntos de datos coloreados por clúster
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, s=50, cmap='viridis')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clustering con K-Means (PCA)")
plt.show()