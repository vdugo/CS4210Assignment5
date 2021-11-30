#-------------------------------------------------------------------------
# AUTHOR: Vincent Verdugo
# FILENAME: clustering.py
# SPECIFICATION: Runs K-means clustering algorithm with k from 2 to 20
# gets the highest silhouette coefficient value of all the runs
# then calculates K-means homogeneity score and agglomerative clustering
# homogeneity score
# FOR: CS 4210 - Assignment #5
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

X_training = []
# dictionary to store mapped values of k to silhouette coefficients
# e.g. 2 => 0.122
sil_coefficients = {}

# reading the data by using Pandas library
df = pd.read_csv('training_data.csv', sep=',', header=None) 

# assign your training data to X_training feature matrix
X_training = df.copy()

#run kmeans testing different k values from 2 until 20 clusters
for k in range(2,21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     score = silhouette_score(X_training, kmeans.labels_)
     sil_coefficients[k] = score

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(list(sil_coefficients.keys()), list(sil_coefficients.values()))
plt.show()

#reading the validation data (clusters) by using Pandas library
df_testing = pd.read_csv('testing_data.csv', header=None, index_col=False)
labels = df_testing[0].to_list()

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
best_k = max(sil_coefficients, key=sil_coefficients.get)
print("Best value of K found:", best_k, "with Silhouette coefficient:", sil_coefficients[best_k])
agg = AgglomerativeClustering(n_clusters=max(sil_coefficients, key=sil_coefficients.get), linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
