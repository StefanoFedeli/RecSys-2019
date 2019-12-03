from matplotlib import pyplot

import Clean.utils_new as utils
import numpy as np
import scipy.sparse as sps
from Clean.External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from Clean.External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Clean.External_Libraries.Notebooks_utils.evaluation_function import evaluate_algorithm


URM_matrix = utils.create_coo("../../../../../Original_dataset/URM.csv")
URM_matrix = URM_matrix.tocsr()

features1 = utils.get_second_column("../../../../../Original_dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../../../Original_dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../../../Original_dataset/ICM_asset.csv")
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../../../Original_dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../../../Original_dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../../../Original_dataset/ICM_asset.csv")
items = items1 + items2 + items3

ones = np.ones(len(features))

n_items = URM_matrix.shape[1]
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()

URM_train, URM_test = train_test_holdout(URM_matrix, train_perc = 0.8)


class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit(self, topK, shrink, normalize, similarity):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
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
recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
recommender.fit(shrink=10, topK=10)
users = utils.get_target_users("../../../../../Original_dataset/target_users.csv")
with open("output.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        recommendations = str(recommender.recommend(user_id, at=10))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(user_id) + ", " + recommendations + "\n")



