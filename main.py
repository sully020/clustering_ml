from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
import pandas as pd

def create_dataframes():
    emails = pd.read_csv('emails.csv')
    diabetes = pd.read_csv('diabetes.csv')
    emails = emails.drop(columns = ['Email No.', 'class_val'])
    diabetes = diabetes.drop(columns = ['class_val'])
    return emails, diabetes


def init_cluster_data():
    emails, diabetes = create_dataframes()
    print("K-Means Clustering -\nE-Mails:")
    init_kclusters(emails)
    print("Diabetes: ")
    init_kclusters(diabetes)
    print("Expectation Maximization -\nE-Mails:")
    init_soft_clusters(emails)
    print("Diabetes: ")
    init_soft_clusters(diabetes)

def init_kclusters(dataset):
    n = input('How many clusters would you like? ')
    kmeans = KMeans(n_clusters = int(n), n_init = 'auto')
    kmeans.fit(dataset)
    silhouette = silhouette_score(dataset, kmeans.labels_)
    print("The silhouette score of the " + str(n) + " clusters was: " +"{:.3f}".format(silhouette) + '\n')

def init_soft_clusters(dataset):
    n = input('How many EM clusters would you like? ')
    exp_max = GaussianMixture(n_components = int(n))
    exp_max.fit(dataset)
    aic = exp_max.aic(dataset)  # Lower = better
    log_likely = exp_max.score(dataset)
    print("The aic score of the " + str(n) + " EM clusters was: " + "{:.3f}".format(aic))
    print("The log-likelihood value of the " + str(n) + " EM clusters was: " + "{:.3f}".format(log_likely) + '\n')


def reduce_dimensions():
    emails, diabetes = create_dataframes()
    email_pca(emails)
    diabetes_pca(diabetes)
    ica_reduction(emails)
    ica_reduction(diabetes)

def email_pca(emails):
    pca = PCA(n_components = 1500, svd_solver = 'randomized')
    new_feature_set = pca.fit_transform(emails)
    print(len(new_feature_set[0]))
    return new_feature_set

def diabetes_pca(diabetes):
    pca = PCA(n_components = 'mle')
    new_feature_set = pca.fit_transform(diabetes)
    print(len(new_feature_set[0]))
    return new_feature_set

def ica_reduction(dataset):
    ica = FastICA()
    new_feature_set = ica.fit_transform(dataset)
    return new_feature_set()

def rando_projection():
    pass
    

def main():
    #init_cluster_data()
    reduce_dimensions()

main()

