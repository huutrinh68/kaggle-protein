from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

for k, (train_index, test_index) in enumerate(msss.split(X, y)):
    print("k:", k, "TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]