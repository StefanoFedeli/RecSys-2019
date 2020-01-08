import utils_new as utils
import numpy as np
import scipy.sparse as sps
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from External_Libraries.Notebooks_utils.evaluation_function import evaluate_algorithm
import matplotlib.pyplot as pyplot
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from External_Libraries.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from External_Libraries.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os
from External_Libraries.DataIO import DataIO
from External_Libraries.Base.Recommender_utils import check_matrix, similarityMatrixTopK
from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from External_Libraries.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
import random as rand
import evaluator
from Reccomenders.NonPersonalised_Reccomenders.TopPop.top_pop import TopPopRecommender as topPop
from Reccomenders.NonPersonalised_Reccomenders.TopPop.top_pop_cluster import TopPopClusterRecommender as topPopCluster
from Reccomenders.NonPersonalised_Reccomenders.TopPop.LightFMTopPop import LightFMTopPopRecommender as lightFMTopPop

URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))

class ItemKNNScoresHybridRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1-self.alpha)

        return item_weights

class ItemCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK, shrink, normalize=True, similarity="jaccard"):
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

itemColl = ItemKNNCFRecommender(URM_all)
itemColl.fit(shrink=50, topK=10)

elasticNet = SLIMElasticNetRecommender(URM_all)
elasticNet.fit()

users = utils.get_target_users("../../../Dataset/target_users.csv")
hybridrecommender = ItemKNNScoresHybridRecommender(URM_all, itemColl, elasticNet)

hybridrecommender.fit(0.3)
with open("../../../Outputs/itemColl+elasticNet_0.3.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(hybridrecommender.recommend(user_id)[:10]) + "\n")

hybridrecommender.fit(0.5)
with open("../../../Outputs/itemColl+elasticNet_0.5.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(hybridrecommender.recommend(user_id)[:10]) + "\n")

hybridrecommender.fit(0.7)
with open("../../../Outputs/itemColl+elasticNet_0.7.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(hybridrecommender.recommend(user_id)[:10]) + "\n")
