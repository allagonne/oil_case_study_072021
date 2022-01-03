import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

path_to_modelize = PATH_TO_MODELIZE ## for import
path_modelized = PATH_MODELIZED ## for export
transactions_fuel = pd.read_csv(path_to_modelize+'table_transactions_fuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal=',')
transactions_nonfuel = pd.read_csv(path_to_modelize+'table_transactions_nonfuel.csv', delimiter = ';',encoding = "ISO-8859-1",decimal=',')

# define dataset
np_fuel = transactions_fuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')
np_nonfuel = transactions_nonfuel[['TR_AMOUNT','TR_COUNT']].to_numpy(dtype='float')
# define the model
model = GaussianMixture(n_components=10)
# fit the model with fuel data
model.fit(np_fuel)
yhat = model.predict(np_fuel)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(np_fuel[row_ix, 0], np_fuel[row_ix, 1])
plt.savefig(path_modelized+'fuel_transactions_GMM1_10comp_fuel.png', dpi=200)
plt.clf()

# fit the model with nonfuel data
model.fit(np_nonfuel)
yhat = model.predict(np_nonfuel)
clusters = np.unique(yhat)
for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	plt.scatter(np_nonfuel[row_ix, 0], np_nonfuel[row_ix, 1])
plt.savefig(path_modelized+'fuel_transactions_GMM2_10comp_nonfuel.png', dpi=200)
plt.clf()