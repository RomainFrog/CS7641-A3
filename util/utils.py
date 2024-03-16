import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RANDOM_SEED = 42
RANDOM_SEEDS = [42, 43, 44, 45, 46]


def learning_curve_with_cross_validation(estimator, X, y, train_sizes, cv=5, scoring=None, dataset_name=None):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
     # if makescorer, get the metric name from the scorer
    if hasattr(scoring, '__call__'):
        scoring = scoring._score_func.__name__
    plt.ylabel(scoring if scoring else "Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")

    plt.legend(loc="best")
    clf_name = str(estimator).split('(')[0]
    if dataset_name:
        plt.savefig(f'figures/{clf_name}_{dataset_name}_learning_curve.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'figures/{clf_name}_learning_curve.pdf', bbox_inches='tight')
    plt.show()


def perform_grid_search(model, X_train, y_train, param_grid, scoring, cv, integer_param=False, show_plot=True, verbose=0, dataset_name=None):

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv, return_train_score=True, n_jobs=-1, verbose=verbose)
    train_scores = grid_search.fit(X_train, y_train)

    # Store results
    results = grid_search.cv_results_
    mean_train_scores = results['mean_train_score']
    mean_test_scores = results['mean_test_score']
    params = results['params']

    print(mean_test_scores)


    # if makescorer, get the metric name from the scorer
    if hasattr(scoring, '__call__'):
        scoring = scoring._score_func.__name__

    if show_plot:
        # Plot the training and validation curves
        plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 18})
        for _, param in enumerate(param_grid.keys()):
            param_values = [params[j][param] for j in range(len(params))]
            plt.plot(param_values, mean_train_scores, marker='o', linestyle='-', color='r', label=f'Training ({param})')
            plt.plot(param_values, mean_test_scores, marker='o', linestyle='-', color='g', label=f'Validation ({param})')

        plt.title('Grid Search Results')
        plt.xlabel(f'Hyperparameter Value ({param})')
        plt.ylabel(f'Mean {scoring}')
        plt.legend()
        plt.grid(True)
        target_feature = list(param_grid.keys())[0]
        if integer_param:
            plt.xticks(np.arange(min(param_values), max(param_values)+1, 1.0))
        if dataset_name:
            plt.savefig(f'figures/{str(model).split("(")[0]}_{dataset_name}_grid_search_{target_feature}.pdf', bbox_inches='tight')
        plt.savefig(f'figures/{str(model).split("(")[0]}_grid_search_{target_feature}.pdf', bbox_inches='tight')
        plt.show()

    # Return the best trained model
    return grid_search.best_estimator_


def best_model_test_set_metrics(model, X_train, y_train, X_test, y_test):
    # print the best hyperparameters in a nice format, one per line
    print('Best hyperparameters:')
    for param_name in sorted(model.get_params().keys()):
        print(f'\t{param_name}: {model.get_params()[param_name]}')
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    print('\n ------------------- \n')
    # Print metrics (accuracy, precision, recall, f1, roc_auc)
    print(f'Accuracy: {model.score(X_test, y_test)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
    print(f'F1: {f1_score(y_test, y_pred, average="weighted")}')
    # print(f'ROC AUC: {roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovr")}')


from sklearn.decomposition import PCA

def plot_pca_elbow(X, pca, dataset_name="Digits"):

    # Fit PCA on the scaled training data
    pca.fit(X)

    # Get eigenvalues
    eigenvalues = pca.explained_variance_

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    # Plot eigenvalues to visualize the explained variance
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='-', linewidth=2, markersize=7)
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalue')
    plt.title(f'PCA Elbow Method on {dataset_name} dataset')
    # Add a gridd to the plot
    plt.grid()
    # x axis ticks to be integers (one every 5)
    plt.xticks(np.arange(0, len(eigenvalues)+1, 5))
    plt.savefig(f'figures/DR/{dataset_name}_pca_elbow.pdf', bbox_inches='tight')
    plt.show()

def plot_pca_cumulative_explained_variance_ratio(X, pca, dataset_name="Digits", ratio_threshold=0.9):
    # Fit PCA on the scaled training data
    pca.fit(X)

    # Get eigenvalues
    eigenvalues = pca.explained_variance_

    # Get the cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    # Plot eigenvalues to visualize the explained variance
    plt.plot(range(1, len(eigenvalues)+1), cumulative_explained_variance, marker='o', linestyle='-')
    # Add a horizontal line at the threshold
    plt.axhline(y=ratio_threshold, color='r', linestyle='--', label=f'{ratio_threshold} threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Cumulative Explained Variance Ratio on {dataset_name} dataset')
    # Add a gridd to the plot
    plt.grid()
    # x axis ticks to be integers (one every 5)
    plt.xticks(np.arange(0, len(eigenvalues)+1, 5))
    plt.savefig(f'figures/DR/{dataset_name}_pca_cumulative_explained_variance_ratio.pdf', bbox_inches='tight')
    plt.show()



from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from tqdm import tqdm

def plot_ica_mean_kurtosis(X, n_min, n_max, dataset_name="Digits"):
    
    mean_kurtosis = []
    n_components = range(n_min, n_max)

    for n in tqdm(n_components):
        ica = FastICA(n_components=n, random_state=RANDOM_SEED, max_iter=1000, tol=0.01)
        X_train_ica = ica.fit_transform(X)
        mean_kurtosis.append(np.mean(kurtosis(X_train_ica)))

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(n_components, mean_kurtosis, marker='', linestyle='-', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Kurtosis')
    plt.title(f'ICA Mean Kurtosis on {dataset_name} dataset')
    plt.grid()
    plt.xticks(np.arange(0, len(n_components)+1, 5))
    plt.savefig(f'figures/DR/{dataset_name}_ica_mean_kurtosis.pdf', bbox_inches='tight')
    plt.show()


def plot_ordered_ica_kurtosis(X, ica, dataset_name="Digits"):
    X_train_ica = ica.fit_transform(X)
    kurts = kurtosis(X_train_ica)


    sorted_indices = np.argsort(kurts)[::-1]
    sorted_kurts = kurts[sorted_indices]  # Sort kurtosis values accordingly
    sorted_dimensions = sorted_indices + 1

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(range(1, len(sorted_kurts)+1), sorted_kurts, linewidth=2, marker='', linestyle='-')
    plt.xlabel('Component')
    plt.ylabel('Kurtosis')
    plt.title(f'ICA Ordered Component Kurtosis on {dataset_name} dataset')
    plt.grid()
    plt.xticks(np.arange(0, len(sorted_kurts)+1, 5))
    plt.savefig(f'figures/DR/{dataset_name}_ica_ordered_kurtosis.pdf', bbox_inches='tight')




from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def k_medoid_multi_seed(X, k_min, k_max, metric="euclidean"):
    # Initialize lists to store results
    kmedoids_train_sil_mean = []
    kmedoids_train_sil_std = []
    kmedoids_train_wcss_mean = []
    kmedoids_train_wcss_std = []
    kmedoids_train_db_mean = []
    kmedoids_train_db_std = []
    kmedoids_train_ch_mean = []
    kmedoids_train_ch_std = []

    # Loop over different numbers of clusters
    for i in tqdm(range(k_min, k_max)):
        silhouette_scores = []
        inertia_scores = []
        bd_scores = []
        ch_scores = []
        for seed in RANDOM_SEEDS:
            # Initialize KMedoids with a specific random seed
            kmedoids = KMedoids(n_clusters=i, init='k-medoids++', max_iter=1000, random_state=seed, metric=metric)
            kmedoids.fit(X)
            # Compute silhouette score for this seed
            silhouette_scores.append(silhouette_score(X, kmedoids.labels_))
            # Compute Davies-Bouldin score for this seed
            bd_scores.append(davies_bouldin_score(X, kmedoids.labels_))
            # Compute Calinski-Harabasz score for this seed
            ch_scores.append(calinski_harabasz_score(X, kmedoids.labels_))
            # Compute inertia for this seed
            inertia_scores.append(kmedoids.inertia_)
        # Calculate mean and standard deviation of silhouette scores for this number of clusters
        mean_silhouette_score = np.mean(silhouette_scores)
        std_silhouette_score = np.std(silhouette_scores)
        kmedoids_train_sil_mean.append(mean_silhouette_score)
        kmedoids_train_sil_std.append(std_silhouette_score)
        # Calculate mean and standard deviation of inertia scores for this number of clusters
        mean_inertia = np.mean(inertia_scores)
        std_inertia = np.std(inertia_scores)
        kmedoids_train_wcss_mean.append(mean_inertia)
        kmedoids_train_wcss_std.append(std_inertia)
        # Calculate mean and standard deviation of Davies-Bouldin scores for this number of clusters
        mean_bd = np.mean(bd_scores)
        std_bd = np.std(bd_scores)
        kmedoids_train_db_mean.append(mean_bd)
        kmedoids_train_db_std.append(std_bd)
        # Calculate mean and standard deviation of Calinski-Harabasz scores for this number of clusters
        mean_ch = np.mean(ch_scores)
        std_ch = np.std(ch_scores)
        kmedoids_train_ch_mean.append(mean_ch)
        kmedoids_train_ch_std.append(std_ch)

    # Convert lists to numpy arrays for easy plotting
    kmedoids_train_sil_mean = np.array(kmedoids_train_sil_mean)
    kmedoids_train_sil_std = np.array(kmedoids_train_sil_std)
    kmedoids_train_wcss_mean = np.array(kmedoids_train_wcss_mean)
    kmedoids_train_wcss_std = np.array(kmedoids_train_wcss_std)
    kmedoids_train_db_mean = np.array(kmedoids_train_db_mean)
    kmedoids_train_db_std = np.array(kmedoids_train_db_std)
    kmedoids_train_ch_mean = np.array(kmedoids_train_ch_mean)
    kmedoids_train_ch_std = np.array(kmedoids_train_ch_std)


    return kmedoids_train_sil_mean, kmedoids_train_sil_std, kmedoids_train_wcss_mean, kmedoids_train_wcss_std, kmedoids_train_db_mean, kmedoids_train_db_std, kmedoids_train_ch_mean, kmedoids_train_ch_std


def plot_k_medoid_multi_seed(k_min, k_max, metric, kmedoids_train_sil_mean, kmedoids_train_sil_std, kmedoids_train_wcss_mean, kmedoids_train_wcss_std):

    # increase font size
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(range(k_min, k_max), kmedoids_train_sil_mean, marker='o', color=color)
    ax1.fill_between(range(k_min, k_max), kmedoids_train_sil_mean - kmedoids_train_sil_std, kmedoids_train_sil_mean + kmedoids_train_sil_std, alpha=0.3, color=color)
    # add vertical line for best silhouette score
    ax1.axvline(x=np.argmax(kmedoids_train_sil_mean) + k_min, color='b', linestyle='--', label='Best Silhouette Score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Inertia', color=color)  
    ax2.plot(range(k_min, k_max), kmedoids_train_wcss_mean, marker='o', color=color)
    ax2.fill_between(range(k_min, k_max), kmedoids_train_wcss_mean - kmedoids_train_wcss_std, kmedoids_train_wcss_mean + kmedoids_train_wcss_std, alpha=0.3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f'KMedoids: Silhouette and Inertia vs number of clusters')
    plt.grid(True)

    plt.savefig(f'figures/CLUSTERING/KMedoids_search_{metric}.pdf', bbox_inches='tight')
    plt.show()



from sklearn.mixture import GaussianMixture
def gaussian_mixture_multi_seed(X, k_min, k_max, covariance_type="full"):
    gmm_train_sil_mean = []
    gmm_train_sil_std = []
    gmm_train_bic_mean = []
    gmm_train_bic_std = []
    gmm_train_aic_mean = []
    gmm_train_aic_std = []

    for i in tqdm(range(k_min, k_max)):
        silhouette_scores = []
        bic_scores = []
        aic_scores = []
        for seed in RANDOM_SEEDS:
            gmm = GaussianMixture(n_components=i, covariance_type=covariance_type, random_state=seed)
            gmm.fit(X)
            silhouette_scores.append(silhouette_score(X, gmm.predict(X)))
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
        mean_silhouette_score = np.mean(silhouette_scores)
        std_silhouette_score = np.std(silhouette_scores)
        gmm_train_sil_mean.append(mean_silhouette_score)
        gmm_train_sil_std.append(std_silhouette_score)
        mean_bic = np.mean(bic_scores)
        std_bic = np.std(bic_scores)
        gmm_train_bic_mean.append(mean_bic)
        gmm_train_bic_std.append(std_bic)
        mean_aic = np.mean(aic_scores)
        std_aic = np.std(aic_scores)
        gmm_train_aic_mean.append(mean_aic)
        gmm_train_aic_std.append(std_aic)

    gmm_train_sil_mean = np.array(gmm_train_sil_mean)
    gmm_train_sil_std = np.array(gmm_train_sil_std)
    gmm_train_bic_mean = np.array(gmm_train_bic_mean)
    gmm_train_bic_std = np.array(gmm_train_bic_std)
    gmm_train_aic_mean = np.array(gmm_train_aic_mean)
    gmm_train_aic_std = np.array(gmm_train_aic_std)

    return gmm_train_sil_mean, gmm_train_sil_std, gmm_train_bic_mean, gmm_train_bic_std, gmm_train_aic_mean, gmm_train_aic_std


def plot_gmm_multi_seed(k_min, k_max, gmm_train_sil_mean, gmm_train_sil_std, gmm_train_bic_mean, gmm_train_bic_std, gmm_train_aic_mean, gmm_train_aic_std, covariance_type):
    
     # increase font size
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(range(k_min, k_max), gmm_train_sil_mean, marker='o', color=color)
    ax1.fill_between(range(k_min, k_max), gmm_train_sil_mean - gmm_train_sil_std, gmm_train_sil_mean + gmm_train_sil_std, alpha=0.3, color=color)
    # add vertical line for best silhouette score
    ax1.axvline(x=np.argmax(gmm_train_sil_mean) + k_min, color='b', linestyle='--', label='Best Silhouette Score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Inertia', color=color)  
    ax2.plot(range(k_min, k_max), gmm_train_bic_mean, marker='o', color=color)
    ax2.fill_between(range(k_min, k_max), gmm_train_bic_mean - gmm_train_bic_std, gmm_train_bic_mean + gmm_train_bic_std, alpha=0.3, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'GMM: Silhouette Score vs AIC and BIC ({covariance_type} covariance)')
    plt.grid(True)
    plt.savefig(f'figures/CLUSTERING/GMM_search_{covariance_type}.pdf', bbox_inches='tight')
    plt.show()