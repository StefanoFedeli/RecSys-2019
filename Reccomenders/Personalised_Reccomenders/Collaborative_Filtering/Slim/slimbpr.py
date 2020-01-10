import scipy.sparse as sps
import time
import numpy as np
import random
import evaluator as evaluate
import utils_new as utils
import External_Libraries.Recommender_utils as mauri

from External_Libraries.Base.BaseRecommender import BaseRecommender as BaseRecommender
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate


class SLIM_BPR_Recommender(BaseRecommender):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM):
        self.URM = URM
        self.numEpoch = 0
        self.learning_rate = 0
        self.epochs = 0

        self.itemPopularity = np.array(URM.sum(axis=0)).squeeze()

        self.n_users = self.URM.shape[0]
        self.n_items = self.URM.shape[1]

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))
        # self.similarity_matrix = sps.csr_matrix(sps.load_npz("../../../../Dataset/similarities/Col-Sim-train.npz")).todense().getA()


        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.URM.indptr[user_id]
            end_pos = self.URM.indptr[user_id + 1]

            if len(self.URM.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        super().__init__(URM)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = self.URM[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False
        neg_item_id = 0
        lanci = 0
        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)
            lanci += 1
            if neg_item_id not in userSeenItems and (self.numEpoch < self.epochs*2/3 or self.itemPopularity[neg_item_id] != 0 or lanci > self.n_items-5):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self, dropoff=0.05):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM.nnz * dropoff)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM[user_id, :].indices

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

            if time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, learning_rate=1e-3, epoch=70, ratio=2.5, limit=120, path="../../../../"):
        self.similarity_matrix = sps.csr_matrix(sps.load_npz(path + "Dataset/similarities/Slim-Sim-train.npz"))
        """
        self.itemPopularity[self.itemPopularity < limit] = 0
        self.learning_rate = learning_rate
        self.epochs = epoch

        for self.numEpoch in range(self.epochs):
            print("Epoch N°" + str(self.numEpoch))
            self.epochIteration(ratio/epoch)

        self.similarity_matrix = self.similarity_matrix.T
        self.similarity_matrix = mauri.similarityMatrixTopK(self.similarity_matrix, verbose=True, k=16)
        sps.save_npz("../../../../Dataset/similarities/Slim-Sim.npz", self.similarity_matrix)
        """

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, at=10,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        result = super().recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag,
                                   remove_custom_items_flag, return_scores)
        if return_scores is True:
            return result
        else:
            return result[:at]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_profile = self.URM[user_id_array]
        if items_to_compute is None:
            return user_profile.dot(self.similarity_matrix).toarray()
        else:
            return user_profile.dot(self.similarity_matrix[items_to_compute]).toarray()


def main_slim():
    URM = sps.csr_matrix(sps.load_npz("../../../../Dataset/data_all.npz"))
    URM_test = sps.csr_matrix(sps.load_npz("../../../../Dataset/data_test.npz"))
    users = utils.get_target_users("../../../../Dataset/target_users.csv", seek=10)
    validator = validate(URM_test, [10])
    recommender = SLIM_BPR_Recommender(URM)
    recommender.fit()

    '''
    for learning_rate in [1e-3]:
        for epocs in [70]:
            for lim in [80,120]:
                recommender = SLIM_BPR_Recommender(URM)
                recommender.fit(learning_rate, epocs)
                for k in [10]:
                    print("LR:{0}, EPOCHS:{1}, DROPOFF:{4:.3f}, LIMIT={3}, k={2}".format(learning_rate, epocs, k, lim,2.5/epocs))
                    recommender.compute_similarity(k)
                    print(evaluate.evaluate(users, recommender, URM_test, 10))
                    results = validator.evaluateRecommender(recommender)
                    #print(results[0])
                    print(results[1])
                    print("\n")
    '''


#main_slim()


