from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_dataframes():
    emails = pd.read_csv('emails.csv')
    diabetes = pd.read_csv('diabetes.csv')
    emails = emails.drop(columns = ['Email No.', 'class_val'])
    diabetes = diabetes.drop(columns = ['class_val'])
    return emails, diabetes


def init_cluster_data():
    print("Clustering\n-------------------------")
    emails, diabetes = create_dataframes()
    print("K-Means Clustering -\nE-Mails:")
    kclusters(emails)
    print("Diabetes: ")
    kclusters(diabetes)
    print("Expectation Maximization -\nE-Mails:")
    init_soft_clusters(emails)
    print("Diabetes: ")
    init_soft_clusters(diabetes)

def kclusters(dataset):
    n = input('How many clusters would you like? ')
    kmeans = KMeans(n_clusters = int(n))
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
    print("Dimension Reduction\n-------------------------")
    emails, diabetes = create_dataframes()
    print('Principal Component Analysis -\nE-Mails:')
    email_pca(emails)
    print("Diabetes: ")
    diabetes_pca(diabetes)
    print('\nIndependent Component Analysis -\nE-Mails:') # Reference properties in paper
    ica_reduction(emails)
    print("Diabetes: ")
    ica_reduction(diabetes)
    print('\nIndependent Component Analysis -\nE-Mails:')
    random_proj(emails)
    print("Diabetes: ")
    random_proj(diabetes)

def email_pca(emails):
    pca = PCA(n_components = 1500, svd_solver = 'randomized')
    new_feature_set = pca.fit_transform(emails)
    return new_feature_set

def diabetes_pca(diabetes):
    pca = PCA(n_components = 'mle')
    new_set = pca.fit_transform(diabetes)
    print(len(new_set[0]))   # Output new number of features calculated using MLE formula (Minka)
    return new_set

def ica_reduction(dataset):
    n = input('How many components would you like? ')
    ica = FastICA(n_components = int(n)) # Check rising
    new_set = ica.fit_transform(dataset)
    print(len(new_set[0]))
    return new_set

def random_proj(dataset):
    n = input('How many components would you like? ')
    projector = GaussianRandomProjection(n_components = int(n))
    new_set = projector.fit_transform(dataset)
    print(len(new_set[0]))
    return new_set


def cluster_new_sets():
    kcluster_pca()
    exp_max_pca()
    kcluster_ica()
    exp_max_ica()
    kcluster_rproj()
    exp_max_rproj()


def kcluster_pca():
    emails, diabetes = create_dataframes()
    emails = email_pca(emails)
    diabetes = diabetes_pca(diabetes)
    kclusters(emails)
    kclusters(diabetes)

def exp_max_pca():
    emails, diabetes = create_dataframes()
    emails = email_pca(emails)
    diabetes = diabetes_pca(diabetes)
    init_soft_clusters(emails)
    init_soft_clusters(diabetes)

def kcluster_ica():
    emails, diabetes = create_dataframes()
    emails = ica_reduction(emails)
    diabetes = ica_reduction(diabetes)
    kclusters(emails)
    kclusters(diabetes)

def exp_max_ica():
    emails, diabetes = create_dataframes()
    emails = ica_reduction(emails)
    diabetes = ica_reduction(diabetes)
    init_soft_clusters(emails)
    init_soft_clusters(diabetes)

def kcluster_rproj():
    emails, diabetes = create_dataframes()
    emails = random_proj(emails)
    diabetes = random_proj(diabetes)
    kclusters(emails)
    kclusters(diabetes)

def exp_max_rproj():
    emails, diabetes = create_dataframes()
    emails = random_proj(emails)
    diabetes = random_proj(diabetes)
    init_soft_clusters(emails)
    init_soft_clusters(diabetes)



def main():
    #init_cluster_data()
    reduce_dimensions()

main()

