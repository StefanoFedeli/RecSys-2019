import scipy.sparse as sps

import utils_new as utils
import numpy as np
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

URM_train = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_train.npz"))
URM_all = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))

features1 = utils.get_second_column("../../../../../Dataset/UCM_age.csv")
features2 = utils.get_second_column("../../../../../Dataset/UCM_region.csv")
features = features1 + features2

entity1 = utils.get_first_column("../../../../../Dataset/UCM_age.csv")
entity2 = utils.get_first_column("../../../../../Dataset/UCM_region.csv")
entities = entity1 + entity2

ones = np.ones(len(features))
UCM_all = sps.coo_matrix((ones, (entities, features)),shape=URM_all.shape)
UCM_all = UCM_all.tocsr()

UCM = sps.coo_matrix((np.ones(len(features1)), (entity1, features1)))
UCM = UCM.tocsr()


class UserCBFKNNRecommender(object):

    def __init__(self, URM, UCM, URM_t):
        self.URM = URM
        self.UCM = UCM
        self.train = URM_t

    def fit(self, topK, shrink, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.UCM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.train.indptr[user_id]
        end_pos = self.train.indptr[user_id + 1]

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


reccomender = UserCBFKNNRecommender(URM_all, UCM_all, URM_train)
reccomender.fit(80, 100)

targetUsers = utils.get_target_users("../../../../../Dataset/target_users.csv")
with open("../../../../../Outputs/CBU.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in targetUsers:
        f.write(str(user_id) + "," + utils.trim(reccomender.recommend(user_id,at=10)) + "\n")
