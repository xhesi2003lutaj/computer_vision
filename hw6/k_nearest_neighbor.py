import numpy as np


def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M, D1 = X1.shape
    N, D2 = X2.shape
    assert D1 == D2, "Feature dimensions of X1 and X2 must match"

    # squared norms of X1 and X2
    X1_squared_norms = np.sum(X1**2, axis=1).reshape(-1, 1)  # Shape (M, 1)
    X2_squared_norms = np.sum(X2**2, axis=1).reshape(1, -1)  # Shape (1, N)

    cross_term = np.dot(X1, X2.T)  

    dists = np.sqrt(X1_squared_norms + X2_squared_norms - 2 * cross_term)

    assert dists.shape == (M, N), f"dists should have shape (M, N), got {dists.shape}"
    # print("pas asert")

    return dists



def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    # Use the distance matrix to find the k nearest neighbors of the ith
    # testing point, and use y_train to find the labels of these
    # neighbors.

    # Once you have found the labels of the k nearest neighbors, you
    # need to find the most common label in the list closest_y of labels.
    # Store this label in y_pred[i]. Break ties by choosing the smaller
    # label.
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=np.int64) 

    for i in range(num_test):
        nearest_neighbors = np.argsort(dists[i])[:k]
        
        closest_y = y_train[nearest_neighbors]
        
        y_pred[i] = np.bincount(closest_y).argmax()

    return y_pred


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Return the folds in this order to match the staff solution!

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    returns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)
        
    hint: you may find np.hstack and np.vstack helpful for this part

    """
    assert X_train.shape[0] == y_train.shape[0]

    X_splits = np.array_split(X_train, num_folds)
    y_splits = np.array_split(y_train, num_folds)

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = []
    y_trains = []
    X_vals = []
    y_vals = []

    for i in range(num_folds):

        X_val = X_splits[i]
        y_val = y_splits[i]
        X_train_fold = np.vstack(X_splits[:i] + X_splits[i + 1:])
        y_train_fold = np.hstack(y_splits[:i] + y_splits[i + 1:])

        X_trains.append(X_train_fold)
        y_trains.append(y_train_fold)
        X_vals.append(X_val)
        y_vals.append(y_val)

    # converting lists to numpy arrays
    X_trains = np.array(X_trains)
    y_trains = np.array(y_trains)
    X_vals = np.array(X_vals)
    y_vals = np.array(y_vals)

    return X_trains, y_trains, X_vals, y_vals

