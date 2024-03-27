Code base is publicly available here: https://github.com/RomainFrog/CS7641-A3

To run the code, first create an environment with the following libraries:
- numpy
- pandas
- sklearn
- sklearn_extra
- umap-learn
- matplotlib
- seaborn
- ucimlrepo
- tqdm

The code is provided with the following architecture:

- util\
	- utils.py  (auxiliary functions for cross validation, hyper parameter search and visualization)
- load_dataset.py (contains function called in notebooks to automatically download and clean datasets)
- titanic_clustering.ipynb (notebook to run all titanic experiments)
- digits_clustering.ipynb (notebook to run all digits experiments)

No additional data is required since Titanic is downloaded from sklearn.datasets and digits from ucimlrepo.

