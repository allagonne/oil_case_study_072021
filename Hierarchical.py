import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

path_to_modelize = PATH_TO_MODELIZE ## for import
path_modelized = PATH_MODELIZED ## for export
transactions_fuel = pd.read_csv(path_to_modelize+'table_transactions_fuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal='.')
transactions_nonfuel = pd.read_csv(path_to_modelize+'table_transactions_nonfuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal='.')

#print(transactions_fuel.head(5))

np_fuel = transactions_fuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')
np_nonfuel = transactions_nonfuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')

## 1) a) plot fuel transactions
plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.1)
plt.scatter(np_fuel[:,0],np_fuel[:,1], label='True Position')
plt.xlabel('amount per day')
plt.ylabel('number of transactions per day')
plt.title('transactions of fuel products')
plt.savefig(path_modelized+'fuel_transactions.png', dpi=200)

## 1) b) plot nonfuel transactions
plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.1)
plt.scatter(np_nonfuel[:,0],np_nonfuel[:,1], label='True Position')
plt.xlabel('amount per day')
plt.ylabel('number of transactions per day')
plt.title('transactions of non-fuel products')
plt.savefig(path_modelized+'nonfuel_transactions.png', dpi=200)

##2) a) dendrogram of fuel transactions
linked = linkage(np_fuel, 'single')
#labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            #labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.savefig(path_modelized+'dendrogram_fuel_transactions.png', dpi=200)

## the code gives a RAM error, because hierarchical clustering is not good for big amount of observations
## in practice, n_obs should be <10000