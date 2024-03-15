import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RANDOM_SEED = 42


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



def 