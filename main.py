""" 
Clustering and Dimensionality Reduction - main.py
Christopher D. Sullivan
Professor Brian O'Neill
4/09/23

The goal of this program is to use clustering and dimension-reduction algorithms in order
to explore and explain previously covered datasets. Namely, the diabetes.csv and
emails.csv are subjected to K-Means clustering, Expectation Maximization, 
PCA, ICA, and Random Projection. Beyond this, the results of these algorithms are
fed to neural networks which attempt to improve on classification of the data
using these newly transformed outputs.

Imports:
https://scikit-learn.org/stable/ - scikit learn
https://pandas.pydata.org/ - pandas
https://docs.python.org/3/library/warnings.html - python warnings

Datasets:
https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv : Balaka Biswas
   - Classifies 5,172 e-mails on whether or not they are considered 'spam'.

https://www.kaggle.com/code/mathchi/diagnostic-a-patient-has-diabetes/notebook : Mehmet Akturk
   - Classifies 768 patients on whether or not they test positive for diabetes.   
"""

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Create pandas dataframes for purpose of clustering data.
def create_dataframes():
    emails = pd.read_csv('emails.csv')
    diabetes = pd.read_csv('diabetes.csv')
    emails = emails.drop(columns = ['Email No.', 'class_val'])
    diabetes = diabetes.drop(columns = ['class_val'])
    return emails, diabetes

# Convenience function used to test initial K-Means and EM Clustering.
def init_cluster_data():
    print("\nClustering\n-------------------------")
    emails, diabetes = create_dataframes()
    print("K-Means Clustering -\nE-Mails:")
    kclusters(emails)
    print("Diabetes: ")
    kclusters(diabetes)
    print("Expectation Maximization -\nE-Mails:")
    soft_cluster(emails)
    print("Diabetes: ")
    soft_cluster(diabetes)

# K-Means Clustering test function - param: dataset
def kclusters(dataset):
    n = input('How many K-Means clusters would you like? ')
    kmeans = KMeans(n_clusters = int(n))
    kmeans.fit(dataset)
    silhouette = silhouette_score(dataset, kmeans.labels_)
    print("The silhouette score of the " + str(n) + " clusters was: " +"{:.3f}".format(silhouette) + '\n')

    clustered_set = kmeans.transform(dataset)
    return clustered_set

# Expectation Maximization test function - param: dataset
def soft_cluster(dataset):
    n = input('How many EM clusters would you like? ')
    exp_max = GaussianMixture(int(n))
    exp_max.fit(dataset)
    aic = exp_max.aic(dataset) # Calculate aic score, lower = better
    log_likely = exp_max.score(dataset)
    print("The aic score of the " + str(n) + " EM clusters was: " + "{:.3f}".format(aic))
    print("The log-likelihood value of the " + str(n) + " EM clusters was: " + "{:.3f}".format(log_likely) + '\n')

    clustered_set = exp_max.predict_proba(dataset)
    return clustered_set


# Convenience function used to test PCA, ICA, Random Proj.
def reduce_dimensions():
    print("Dimension Reduction\n-------------------------")
    emails, diabetes = create_dataframes()
    print('Principal Component Analysis - \nE-mails: ')
    email_pca(emails)
    print("Diabetes: ")
    diabetes_pca(diabetes)
    print('\nIndependent Component Analysis - \nE-mails: ')
    ica_reduction(emails)
    print('Diabetes: ')
    ica_reduction(diabetes)
    print('\nGaussian Random Projection - \nE-mails: ')
    random_proj(emails)
    print('Diabetes: ')
    random_proj(diabetes)
    print('Dimensional Reduction Successfully Completed.')

# emails.csv specific PCA test function
def email_pca(emails):
    n = input('How many PCA components would you like to test? ')
    pca = PCA(n_components = int(n), svd_solver = 'randomized')
    new_set = pca.fit_transform(emails)
    print("Descending Variance Array: " + str(pca.explained_variance_)) # Output eigenvalues of each component in descending order.
    return new_set

# diabetes.csv specific PCA test function
def diabetes_pca(diabetes):
    n = input('How many PCA components would you like to test? ')
    pca = PCA(n_components = int(n))
    new_set = pca.fit_transform(diabetes)
    print("Descending Variance Array: " + str(pca.explained_variance_))
    return new_set                                                                              

# General ICA test function - param: dataset
def ica_reduction(dataset):
    n = input('How many ICA components would you like to test? ')
    ica = FastICA(n_components = int(n), max_iter = 275)
    new_set = ica.fit_transform(dataset)
    print("The ICA algorithm converged in: " + str(ica.n_iter_) + " iterations.")
    return new_set

# General Random Projection test function - param: dataset
def random_proj(dataset):
    n = input('How many Random Projection components would you like to test? ')
    projector = GaussianRandomProjection(n_components = int(n))
    new_set = projector.fit_transform(dataset)
    return new_set


# Convenience function used to test each clustering algorithm on output of all reduction algos.
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

# Test K-Means on both PCA reductions.
def kcluster_pca():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = email_pca(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = diabetes_pca(diabetes)
    kclusters(diabetes)

# Test EM on both PCA reductions.
def exp_max_pca():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = email_pca(emails)
    soft_cluster(emails)
    print("Diabetes: ")
    diabetes = diabetes_pca(diabetes)
    soft_cluster(diabetes)

# Test K-Means on both ICA reductions.
def kcluster_ica():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = ica_reduction(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = ica_reduction(diabetes)
    kclusters(diabetes)

# Test EM on both ICA reductions.
def exp_max_ica():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = ica_reduction(emails)
    soft_cluster(emails)
    print("Diabetes: ")
    diabetes = ica_reduction(diabetes)
    soft_cluster(diabetes)

# Test K-Means on both RP reductions.
def kcluster_rproj():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = random_proj(emails)
    kclusters(emails)
    print("Diabetes: ")
    diabetes = random_proj(diabetes)
    kclusters(diabetes)

# Test EM on both RP reductions.
def exp_max_rproj():
    emails, diabetes = create_dataframes()
    print("E-Mails: ")
    emails = random_proj(emails)
    soft_cluster(emails)
    print("Diabetes: ")
    diabetes = random_proj(diabetes)
    soft_cluster(diabetes)


# Helper function used to split reduced feature set into training and testing data.
def process_reductions(reduced):
    diabetes = pd.read_csv('diabetes.csv')
    y = diabetes['class_val']
    x_train, x_test, y_train, y_test = train_test_split(reduced, y)
    return x_train, x_test, y_train, y_test

# Convenience function used to train and test neural nets on all reduced diabetes datasets.
def test_neural_nets():
    print('Neural Network Training - Reduced Featuresets\n-------------------------\nPCA:')
    pca_diabetes_nn()
    print("ICA:")
    ica_diabetes_nn()
    print('Random Projection: ')
    random_proj_diabetes_nn()
    print('\nK-Means Derived Features:')
    kmeans_diabetes_nn()
    print('\nEM Derived Features:')
    em_diabetes_nn()

nn = MLPClassifier() # Global Neural Network classifier used across test functions.
diabetes = create_dataframes()[1]

# Run neural network on PCA-reduced diabetes.csv
def pca_diabetes_nn():
    reduced = diabetes_pca(diabetes)
    x_train, x_test, y_train, y_test = process_reductions(reduced)
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    print("The PCA-data-trained neural network was " + "{:.2f}".format(train_score) + "% accurate in training.")
    acc = nn.score(x_test, y_test)
    print("The PCA-data-trained neural network was " + "{:.2f}".format(acc) + "% accurate in testing.")

# Run neural network on ICA-reduced diabetes.csv
def ica_diabetes_nn():
    reduced = ica_reduction(diabetes)
    x_train, x_test, y_train, y_test = process_reductions(reduced)
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    print("The ICA-data-trained neural network was " + "{:.2f}".format(train_score) + "% accurate in training.")
    acc = nn.score(x_test, y_test)
    print("The ICA-data-trained neural network was " + "{:.2f}".format(acc) + "% accurate in testing.")

# Run neural network on Randomly Projected diabetes.csv
def random_proj_diabetes_nn():
    reduced = random_proj(diabetes)
    x_train, x_test, y_train, y_test = process_reductions(reduced)
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    print("The Random Projection data neural network was " + "{:.2f}".format(train_score) + "% accurate in training.")
    acc = nn.score(x_test, y_test)
    print("The Random Projection data neural network was " + "{:.2f}".format(acc) + "% accurate in testing.")

# Run neural network on K-Means clustered diabetes.csv
def kmeans_diabetes_nn():
    reduced = kclusters(diabetes)
    x_train, x_test, y_train, y_test = process_reductions(reduced)
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    print("The K-Means neural network was " + "{:.2f}".format(train_score) + "% accurate in training.")
    acc = nn.score(x_test, y_test)
    print("The K-Means neural network was " + "{:.2f}".format(acc) + "% accurate in testing.")

# Run neural network on EM-clustered diabetes.csv
def em_diabetes_nn():
    reduced = soft_cluster(diabetes)
    x_train, x_test, y_train, y_test = process_reductions(reduced)
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    print("The soft-clustered neural network was " + "{:.2f}".format(train_score) + "% accurate in training.")
    acc = nn.score(x_test, y_test)
    print("The soft-clustered neural network was " + "{:.2f}".format(acc) + "% accurate in testing.")


# main function used to test all convenience functions independently.
def main():
    init_cluster_data()
    reduce_dimensions()
    cluster_new_sets()
    test_neural_nets()

main()

