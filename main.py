from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from sklearn import random_projection
import pandas as pd

def create_dataframes():
    emails = pd.read_csv('emails.csv')
    diabetes = pd.read_csv('diabetes.csv')
    emails = emails.drop(columns = ['Email No.', 'class_val'])
    diabetes = diabetes.drop(columns = ['class_val'])
    return emails, diabetes

def init_cluster_data():
    emails, diabetes = create_dataframes()
    init_kcluster_emails(emails)
    init_kcluster_diabetes(diabetes)

def init_kcluster_emails(emails):
    n = input('How many e-mail clusters would you like? ')
    e_kmeans = KMeans(n_clusters = int(n), n_init = 'auto')
    e_kmeans.fit(emails)
    silhouette = silhouette_score(emails, e_kmeans.labels_)
    print("The silhouette score of the " + str(n) + " e-mail clusters was: " +"{:.3f}".format(silhouette))

def init_kcluster_diabetes(diabetes):
    n = input('How many diabetes patient clusters would you like? ')
    d_kmeans = KMeans(n_clusters = int(n), n_init = 'auto')
    d_kmeans.fit(diabetes)
    silhouette = silhouette_score(diabetes, d_kmeans.labels_)
    print("The silhouette score of the " + str(n) + " patient clusters was: " +"{:.3f}".format(silhouette))

init_cluster_data()
