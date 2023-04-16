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
    print('Commencing Principal Component Analysis - \nE-mails: ')
    email_pca(emails)
    print("Diabetes: ")
    diabetes_pca(diabetes)
    print('\nUndergoing Independent Component Analysis - \nE-mails: ') # Reference properties in paper
    ica_reduction(emails)
    print('Diabetes: ')
    ica_reduction(diabetes)
    print('\nRunning Gaussian Random Projection - \nE-mails: ')
    random_proj(emails)
    print('Diabetes: ')
    random_proj(diabetes)
    print('Dimension Reduction Successfully Completed.')

def email_pca(emails):
    n = input('How many PCA components would you like? ')
    pca = PCA(n_components = int(n), svd_solver = 'randomized')
    new_set = pca.fit_transform(emails)
    return new_set

def diabetes_pca(diabetes):
    n = input('How many PCA components would you like? ')
    pca = PCA(n_components = int(n))
    new_set = pca.fit_transform(diabetes)
    return new_set                                                                              

def ica_reduction(dataset):
    n = input('How many ICA components would you like? ')
    ica = FastICA(n_components = int(n)) # Check rising
    new_set = ica.fit_transform(dataset)
    return new_set

def random_proj(dataset):
    n = input('How many Random Projection components would you like? ')
    projector = GaussianRandomProjection(n_components = int(n))
    new_set = projector.fit_transform(dataset)
    return new_set


def cluster_new_sets():
    print('\nClustering of new feature sets\n-------------------------')
    print('PCA-Reduced K-Means - ')
    kcluster_pca()
    print('PCA-Reduced EM - ')
    exp_max_pca()
    print('ICA-Reduced K-Means - ')
    kcluster_ica()
    print('ICA-Reduced EM - ')
    exp_max_ica()
    print('Random-Projection Reduced K-Means -')
    kcluster_rproj()
    print('Random-Projection Reduced EM -')
    exp_max_rproj()

def kcluster_pca():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = email_pca(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = diabetes_pca(diabetes)
    kclusters(diabetes)

def exp_max_pca():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = email_pca(emails)
    init_soft_clusters(emails)
    print("Diabetes: ")
    diabetes = diabetes_pca(diabetes)
    init_soft_clusters(diabetes)

def kcluster_ica():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = ica_reduction(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = ica_reduction(diabetes)
    kclusters(diabetes)

def exp_max_ica():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = ica_reduction(emails)
    init_soft_clusters(emails)
    print("Diabetes: ")
    diabetes = ica_reduction(diabetes)
    init_soft_clusters(diabetes)

def kcluster_rproj():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = random_proj(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = random_proj(diabetes)
    kclusters(diabetes)

def exp_max_rproj():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = random_proj(emails)
    init_soft_clusters(emails)
    print("Diabetes: ")
    diabetes = random_proj(diabetes)
    init_soft_clusters(diabetes)


def test_neural_nets():
    pass

def pca_diabetes_nn():
    pass

def ica_diabetes_nn():
    pass

def random_proj_diabetes_nn():
    pass

def kmeans_diabetes_nn():
    pass

def em_diabetes_nn():
    pass


def main():
    init_cluster_data()
    #reduce_dimensions()
    cluster_new_sets()
    test_neural_nets()

main()

