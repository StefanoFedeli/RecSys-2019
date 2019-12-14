import utils_new as utils
import numpy as np
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sps

URM_train = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_train.npz"))
URM_all = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))

class UserCFKNNRecommender(object):

    def __init__(self, URM, URM_train):
        self.URM = URM
        self.test = URM_train

    def fit(self, topK=50, shrink=100, normalize=True, similarity="jaccard"):
        similarity_object = Compute_Similarity_Python(self.URM.T, shrink=shrink,
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
        start_pos = self.test.indptr[user_id]
        end_pos = self.test.indptr[user_id + 1]

        user_profile = self.test.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

'''
x_tick = [500, 600, 750, 1000, 1200]
MAP_per_k = []

for topK in x_tick:
    recommender = UserCFKNNRecommender(URM_train)
    recommender.fit(shrink=0.0, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("topk.png")


x_tick = [0, 10, 50, 100, 200, 500]
MAP_per_shrinkage = []

for shrink in x_tick:
    recommender = UserCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=600)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])


pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")


'''
recommender = UserCFKNNRecommender(URM_all,URM_train)
recommender.fit(shrink=50, topK=10)
#users = utils.get_target_users("../../../../../Dataset/users_clusters/Coll_U.csv")
users = utils.get_target_users("../../../../../Dataset/target_users.csv")
with open("../../../../../Outputs/Coll_U.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + "," + utils.trim(recommender.recommend(user_id, at=10)) + "\n")
