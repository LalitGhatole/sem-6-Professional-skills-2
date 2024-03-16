# P8: CLUSTERING MODEL 
# a. Clustering algorithms for unsupervised classification. 
# b. Plot the cluster data using matplotlib visualizations. 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('prac8/iris.csv')

# Name the dataset
df.name = "Clustering_Dataset"

# Define features (X)
X = df.drop(columns=['variety'])


# Choose the number of clusters (K)
num_clusters = 3

# Initialize the KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the dataframe
df['cluster'] = cluster_labels

# Plot the clustered data
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster
for cluster in range(num_clusters):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['petal.length'], cluster_data['petal.width'], label=f'Cluster {cluster}')


# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', s=100, label='Centroids')

plt.xlabel('petal.length')
plt.ylabel('petal.width')
plt.title('Clustered Data')
plt.legend()
plt.grid(True)
plt.show()
