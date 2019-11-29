from matplotlib import pyplot

import utils
import numpy as np
from sklearn import preprocessing
import scipy.sparse as sps
from Notebooks_utils.data_splitter import train_test_holdout
from Notebooks_utils.evaluation_function import evaluate_algorithm
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

URM_matrix = utils.create_coo("../../dataset/data_train.csv")
URM_matrix = URM_matrix.tocsr()

URM_train, URM_test = train_test_holdout(URM_matrix, train_perc = 0.8)


class ItemCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="jaccard"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

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
x_tick = [1000, 1500, 2000]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=0.0, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("1.png")


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
pyplot.savefig("2.png")
'''

recommender = ItemCFKNNRecommender(URM_train)
recommender.fit(shrink=50, topK=1000)

users = utils.get_first_column("../../dataset/data_target_users_test.csv")

with open("output.csv", 'w') as f:
    for user_id in users:
        recommendations = str(recommender.recommend(user_id, at=10))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(user_id) + ", " + recommendations + "\n")