import utils_new as utils
import numpy as np
import scipy.sparse as sps
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

features1 = utils.get_second_column("../../../../../Dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../../../Dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../../../Dataset/ICM_asset.csv")
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../../../Dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../../../Dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../../../Dataset/ICM_asset.csv")
items = items1 + items2 + items3

ones = np.ones(len(features))

URM_train = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))
URM_all = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))

n_items = URM_train.shape[1]
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()


class ItemCBFKNNRecommender(object):

    def __init__(self, URM, URM_t, ICM):
        self.URM = URM
        self.ICM = ICM
        self.train = URM_t

    def fit(self, topK, shrink, similarity, normalize = True):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
        sps.save_npz("../../../../../Dataset/CB-Sim.npz", self.W_sparse)
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
x_tick = [10, 50, 100, 200, 500]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
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
    recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
    recommender.fit(shrink=shrink, topK=10)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")

'''
recommender = ItemCBFKNNRecommender(URM_all,URM_train,ICM_all)
recommender.fit(shrink=10, topK=21, similarity="jaccard")
users = utils.get_target_users("../../../../../Dataset/users_clusters/CBI.csv")
with open("../../../../../Outputs/CBI.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(recommender.recommend(user_id, at=10)) + "\n")


