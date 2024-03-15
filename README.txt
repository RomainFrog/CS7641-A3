To run the code, first create an environment with the following libraries:
- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- torch
- ucimlrepo

The code is provided with the following architecture:

- util\
	- utils.py  (auxiliary functions for cross validation, hyper parameter seach and visualization)
- load_dataset.py (contains function called in notebooks to automatically download and clean datasets)
- titanic.ipynb (notebook to run all titanic experiments)
- digits.ipynb (notebook to run all digits experiments)

No additional data is required since Titanic is downloaded from sklearn.datasets and digits from ucimlrepo.

