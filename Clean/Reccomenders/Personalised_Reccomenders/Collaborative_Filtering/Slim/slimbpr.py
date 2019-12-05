# %% [code]
import random
from scipy import stats
from scipy.optimize import fmin
import scipy.sparse as sps
import time
import numpy as np
import pandas as pd

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
URM = sps.load_npz("../input/datasetrecsys/data_all.npz")
URM = URM.tocsr()
URM_mask = URM.copy()
URM_mask.eliminate_zeros()
URM_mask

# %% [code]
n_users = URM_mask.shape[0]
n_items = URM_mask.shape[1]

eligibleUsers = []

for user_id in range(n_users):

    start_pos = URM_mask.indptr[user_id]
    end_pos = URM_mask.indptr[user_id + 1]

    if len(URM_mask.indices[start_pos:end_pos]) > 0:
        eligibleUsers.append(user_id)
eligibleUsers


# %% [code]
def sampleTriplet():
    # By randomly selecting a user in this way we could end up
    # with a user with no interactions
    # user_id = np.random.randint(0, n_users)

    user_id = np.random.choice(eligibleUsers)

    # Get user seen items and choose one
    userSeenItems = URM_mask[user_id, :].indices
    pos_item_id = np.random.choice(userSeenItems)

    negItemSelected = False

    # It's faster to just try again then to build a mapping of the non-seen items
    while (not negItemSelected):
        neg_item_id = np.random.randint(0, n_items)

        if (neg_item_id not in userSeenItems):
            negItemSelected = True

    return user_id, pos_item_id, neg_item_id


# %% [code]
similarity_matrix = np.zeros((n_items, n_items))

user_id, posItem, negItem = sampleTriplet()
userSeenItems = URM_mask[user_id, :].indices

print(posItem)
x_i = similarity_matrix[posItem, userSeenItems]
print(x_i)
x_i = similarity_matrix[posItem, userSeenItems].sum()
x_j = similarity_matrix[negItem, userSeenItems].sum()

print("x_i is {:.2f}, x_j is {:.2f}".format(x_i, x_j))


# %% [code]

def similarityMatrixTopK(item_weights, forceSparseOutput=True, k=120, verbose=False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

        for item_idx in range(nitems):
            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx + 1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


# %% [code]
class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM):
        self.URM = URM

        self.similarity_matrix = np.zeros((n_items, n_items))

        self.URM_mask = self.URM.copy()
        self.URM_mask.eliminate_zeros()

        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz * 0.05)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, userSeenItems] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, userSeenItems] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, learning_rate=1e-3, epochs=100):

        self.learning_rate = learning_rate
        self.epochs = epochs

        for numEpoch in range(self.epochs):
            print("Epoch N°" + str(numEpoch))
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, verbose=True, k=50)

    def recommend(self, user_id, at=10, exclude_seen=True):
        if not isinstance(user_id, int):
            print("eRR")
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


# %% [code]
recommender = SLIM_BPR_Recommender(URM)
recommender.fit()

# %% [code]
validation = pd.read_csv("../input/datasetrecsys/target_users_other.csv", skiprows=1, names=["user"])
validation.head()

# %% [code]
with open("output.csv", "w") as f:
    count = 0
    for row in validation.itertuples():
        if count % 500 == 0:
            print("Elaborated {} users on {}".format(count, validation.user.count))
        recommendations = str(recommender.recommend(row.user))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(row.user) + ", " + recommendations + "\n")
        count += 1

# %% [code]
validation = pd.read_csv("../input/datasetrecsys/target_users.csv", skiprows=1, names=["user"])
validation.head()

# %% [code]
with open("output_all.csv", "w") as f:
    count = 0
    for row in validation.itertuples():
        if count % 500 == 0:
            print("Elaborated {} users on {}".format(count, validation.user.count))
        recommendations = str(recommender.recommend(row.user))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(row.user) + ", " + recommendations + "\n")
        count += 1