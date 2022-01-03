import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

path_to_modelize = PATH_TO_MODELIZE ## for import
path_modelized = PATH_MODELIZED ## for export
transactions_fuel = pd.read_csv(path_to_modelize+'table_transactions_fuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal=',')
transactions_nonfuel = pd.read_csv(path_to_modelize+'table_transactions_nonfuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal=',')

#print(transactions_fuel.head(5))

np_fuel = transactions_fuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')
np_nonfuel = transactions_nonfuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')

## silhouette score, in practice, hard to calculate because of the amount of data
#range_n_clusters = list(range(2,11))
#for n_clusters in tqdm(range_n_clusters):
#    clusterer = KMeans(n_clusters=n_clusters)
#    pred = clusterer.fit_predict(np_fuel[:10000])
#    centers = clusterer.cluster_centers_

#    score = silhouette_score(np_fuel[:10000], pred[:10000])
#    print(f"For n_clusters = {n_clusters}, the silhouette score is {score})")

##1) a) KMeans on fuel data : first attempt with 3 clusters
## init points based on plot
init = np.array([[500000, 5000],
                [200000, 2000],
                [0, 0]],
                np.float64)
model_randominit = KMeans(
    init="random",
    n_clusters=3,
    #n_init=1,
    #max_iter=100,
    random_state=0)\
    .fit(np_fuel)
print(model_randominit.cluster_centers_)
#print(kmeans.n_iter_)
model_default = KMeans(3).fit(np_fuel)
print(model_default.cluster_centers_)

## 1) b) plot the points and the centroids depending on the cluster
## use of tqdm to see the advancement
# takes a long time with full data ~2h so I shortened the size of np_fuel and take random values
for point in tqdm(np_fuel[np.random.choice(np_fuel.shape[0], size=1000, replace=False),:]):
  if model_default.predict(point.reshape(1,-1)) == [0]:
    plt.scatter(point[0], point[1], marker='.', c='b')
  elif model_default.predict(point.reshape(1,-1)) == [1]:
    plt.scatter(point[0], point[1], marker='.',c='g')
  elif model_default.predict(point.reshape(1,-1)) == [2]:
    plt.scatter(point[0], point[1], marker='.', c='r')
for center in model_default.cluster_centers_:
  plt.scatter(center[0],center[1], marker='x')
plt.savefig(path_modelized+'fuel_transactions_kmeans1_reduceddata.png', dpi=200)
plt.clf()

## 2) determine the best number of clusters with the elbow method
kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 100,
   "random_state": 0,
}
## 2) a) compute the residual sum of squares for k from 1 to 8 clusters for fuel data
rss_fuel = []
for k in range(1, 8):
   kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   kmeans.fit(np_fuel)
   rss_fuel.append(kmeans.inertia_)
print(rss_fuel)
## Do the same for nonfuel data
rss_nonfuel = []
for k in range(1, 8):
   kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   kmeans.fit(np_nonfuel)
   rss_nonfuel.append(kmeans.inertia_)
print(rss_nonfuel)
## 2) b) plot the elbow curve
#plt.style.use("fivethirtyeight")
plt.plot(range(1, 8), rss_fuel)
plt.xticks(range(1, 8))
plt.xlabel("Number k of Clusters")
plt.ylabel("sse")
plt.savefig(path_modelized+'fuel_transactions_kmeans2_fuel_elbow.png', dpi=200)
plt.clf()
plt.plot(range(1, 8), rss_nonfuel)
plt.xticks(range(1, 8))
plt.xlabel("Number k of Clusters")
plt.ylabel("sse")
plt.savefig(path_modelized+'fuel_transactions_kmeans3_nonfuel_elbow.png', dpi=200)
plt.clf()
## 2) c) determine the best k with the knee locator
kl_fuel = KneeLocator(range(1, 8), rss_fuel, curve="convex", direction="decreasing") ## gives 2 as the best cluster number
print(f'best number of clusters for fuel data according to the elbow method: {kl_fuel.elbow}')
kl_nonfuel = KneeLocator(range(1, 8), rss_nonfuel, curve="convex", direction="decreasing") ## gives 3 as the best cluster number
print(f'best number of clusters for nonfuel data according to the elbow method: {kl_nonfuel.elbow}')

## 3) a) apply kmeans with kl.elbow for fuel data
model_best_fuel = KMeans(kl_fuel.elbow).fit(np_fuel)
print(f'the centers are {model_best_fuel.cluster_centers_}')

yhat = model_best_fuel.predict(np_fuel)
clusters_fuel = np.unique(yhat)
for cluster in clusters_fuel:
	row_ix = np.where(yhat == cluster)
	plt.scatter(np_fuel[row_ix, 0], np_fuel[row_ix, 1])
plt.savefig(path_modelized+'fuel_transactions_kmeans4_bestmodel_fuel.png', dpi=200)
plt.clf()

## 3) b) apply kmeans with 4 clusters for fuel data, according to visual interpretation of sse elbow curve
model_best_fuel_4 = KMeans(4).fit(np_fuel)
print(f'the centers are {model_best_fuel_4.cluster_centers_}')

yhat = model_best_fuel_4.predict(np_fuel)
clusters_fuel_4 = np.unique(yhat)
for cluster in clusters_fuel_4:
	row_ix = np.where(yhat == cluster)
	plt.scatter(np_fuel[row_ix, 0], np_fuel[row_ix, 1])
plt.savefig(path_modelized+'fuel_transactions_kmeans5_model_fuel_4clusters.png', dpi=200)
plt.clf()

## 3) b) apply kmeans with kl.elbow for non fuel data
model_best_nonfuel = KMeans(kl_nonfuel.elbow).fit(np_nonfuel)
print(f'the centers are {model_best_nonfuel.cluster_centers_}')

yhat = model_best_nonfuel.predict(np_nonfuel)
clusters_nonfuel = np.unique(yhat)
for cluster in clusters_nonfuel:
	row_ix = np.where(yhat == cluster)
	plt.scatter(np_nonfuel[row_ix, 0], np_nonfuel[row_ix, 1])
plt.savefig(path_modelized+'fuel_transactions_kmeans6_bestmodel_nonfuel.png', dpi=200)
plt.clf()