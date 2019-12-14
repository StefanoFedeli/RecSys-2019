import utils_new as utils
import numpy as np
import scipy.sparse as sps
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

URM_train = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_train.npz"))
URM_all = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))

class ItemCFKNNRecommender(object):

    def __init__(self, URM, URM_t):
        self.URM = URM
        self.train = URM_t

    def fit(self, topK, shrink, normalize=True, similarity="jaccard"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
        sps.save_npz("../../../../../Dataset/Col-Sim.npz", self.W_sparse)

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
        start_pos = self.train.indptr[user_id]
        end_pos = self.train.indptr[user_id + 1]

        user_profile = self.train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

'''
x_tick = [500, 750, 1000]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
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
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=1000)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])


pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")
'''


recommender = ItemCFKNNRecommender(URM_all,URM_train)
recommender.fit(shrink=50, topK=10)

#users = utils.get_target_users("../../../../../Dataset/users_clusters/Coll_I.csv")
users = utils.get_target_users("../../../../../Dataset/target_users.csv")
with open("../../../../../Outputs/Coll_I.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(recommender.recommend(user_id, at=10)) + "\n")

