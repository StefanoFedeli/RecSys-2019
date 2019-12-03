import scipy.sparse as sps

import Clean.utils_new as utils
from matplotlib import pyplot
import numpy as np
from Clean.External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from Clean.External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Clean.External_Libraries.Notebooks_utils.evaluation_function import evaluate_algorithm


URM_matrix = utils.create_coo("../../../../../Dataset/URM.csv")
URM_matrix = URM_matrix.tocsr()

features1 = utils.get_second_column("../../../../../Dataset/UCM_age.csv")
features2 = utils.get_second_column("../../../../../Dataset/UCM_region.csv")
features = features1 + features2

entity1 = utils.get_first_column("../../../../../Dataset/UCM_age.csv")
entity2 = utils.get_first_column("../../../../../Dataset/UCM_region.csv")
entities = entity1 + entity2

ones = np.ones(len(features))
UCM_all = sps.coo_matrix((ones, (entities, features)))
UCM_all = UCM_all.tocsr()

UCM = sps.coo_matrix((np.ones(len(features1)), (entity1, features1)))
UCM = UCM.tocsr()
print(UCM)

URM_train, URM_test = train_test_holdout(URM_matrix, train_perc = 0.8)

class UserCBFKNNRecommender(object):

    def __init__(self, URM, UCM):
        self.URM = URM
        self.UCM = UCM

    def fit(self, topK, shrink, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.UCM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)
        self.W_sparse = similarity_object.compute_similarity()
        print(self.W_sparse.data)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

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


reccomender = UserCBFKNNRecommender(URM_train, UCM)
reccomender.fit(0, 10)