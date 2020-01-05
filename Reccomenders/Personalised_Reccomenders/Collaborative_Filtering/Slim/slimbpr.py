import scipy.sparse as sps
import time
import numpy as np
import random
import External_Libraries.Recommender_utils as mauri
import External_Libraries.Notebooks_utils.evaluation_function as evaluate

URM = sps.load_npz("../../../../Dataset/data_train.npz")
URM = URM.tocsr()
URM_mask = URM.copy()
URM_mask.eliminate_zeros()
n_users = URM_mask.shape[0]
n_items = URM_mask.shape[1]


class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM):
        self.URM = URM
        self.numEpoch = 0

        self.itemPopularity = np.array((URM > 0).sum(axis=0)).squeeze()
        self.itemPopularity[self.itemPopularity < 180] = 0

        self.similarity_matrix = np.zeros((n_items, n_items))
        self.similarity_matrix_K = self.similarity_matrix
        #self.similarity_matrix = sps.csr_matrix(sps.load_npz("../../../../Dataset/similarities/Col-Sim-train.npz")).todense().getA()

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
        neg_item_id = 0
        lanci = 0
        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, n_items)
            lanci += 1
            if neg_item_id not in userSeenItems and (self.numEpoch < self.epochs/2 or self.itemPopularity[neg_item_id] != 0 or lanci > n_items-5):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self, dropoff = 0.05):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz * dropoff)

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

            '''if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()
            '''

    def fit(self, learning_rate=1e-2, epochs=48, dropoff=0.05):

        self.learning_rate = learning_rate
        self.epochs = epochs

        for self.numEpoch in range(self.epochs):
            # print("Epoch N°" + str(self.numEpoch))
            self.epochIteration(dropoff)

        self.similarity_matrix = self.similarity_matrix.T

    def compute_similarity(self,k=25):
        self.similarity_matrix_K = mauri.similarityMatrixTopK(self.similarity_matrix, verbose=True, k=k)

    def recommend(self, user_id, at=10, exclude_seen=True):
        if not isinstance(user_id, int):
            print("eRR")
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix_K).toarray().ravel()

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


for learning_rate in [1e-4,1e-2]:
    for epocs in [50,70]:
        for dropoff in [0.007,0.009,0.02,0.04,0.06]:
            recommender = SLIM_BPR_Recommender(URM)
            recommender.fit(learning_rate,epocs,dropoff)
            for k in [15,50]:
                print("LR:{0}, EPOCHS:{1}, DROPOFF:{2}, k={3}".format(learning_rate,epocs,dropoff,k))
                recommender.compute_similarity(k)

                print(evaluate.evaluate_algorithm(sps.load_npz("../../../../Dataset/data_test.npz"), recommender, 10))
                print("\n")