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

def create_clusters():
    user_list = list(range(0, 30911))
    users_dict = dict()

    interaction_age, user_age, age = utils.create_tuples("../../../Dataset/UCM_age.csv",14)
    interaction_region, user_region, region = utils.create_tuples("../../../Dataset/UCM_region.csv",14)
    counter = 0
    interactions = interaction_age + interaction_region
    users = user_age + user_region
    features = age + region
    for u in user_list:
        try:
            index = user_age.index(u)
            target_age = age[index]
        except ValueError:
            target_age = -1
        try:
            index = user_region.index(u)
            target_region = region[index]
        except ValueError:
            target_region = -1

        if target_age>=0 and target_region>=0:
            key = int(str(target_age)+str(target_region))
            if key not in users_dict:
                users_dict[key] = list()
            users_dict[key].append(u)
            user_list.remove(u)
        elif target_age<0 and target_region>=0:
            for i in range(0, max(age)):
                key = int(str(i)+str(target_region))
                if key not in users_dict:
                    users_dict[key] = list()
                users_dict[key].append(u)
            user_list.remove(u)
        elif target_age>=0 and target_region<0:
            for i in range(min(region), max(region)):
                key = int(str(target_age)+str(i))
                if key not in users_dict:
                    users_dict[key] = list()
                users_dict[key].append(u)
            user_list.remove(u)

    users_dict[0] = user_list
    print(users_dict)

    clusterSamples = []
    for key in users_dict:
        clusterSamples.append(users_dict[key][5:15])

    return clusterSamples




URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))

features1 = utils.get_second_column("../../../Dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../Dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../Dataset/ICM_asset.csv")
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../Dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../Dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../Dataset/ICM_asset.csv")
items = items1 + items2 + items3

ones = np.ones(len(features))

n_items = URM_all.shape[1]
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()

clusterSamples = create_clusters()
maxIndex = len(clusterSamples)
users = range(0, 30911)

MAP_itemKNNCF_per_group = []
MAP_userKNNCF_per_group = []
MAP_itemKNNCBF_per_group = []
MAP_pureSVD_per_group = []
MAP_elasticNet_per_group = []
MAP_sslim_per_group = []
MAP_topPop_per_group = []
cutoff = 10

itemKNNCF = ItemKNNCFRecommender(URM_train)
itemKNNCF.fit(topK=10, shrink=50, similarity="jaccard")

userKNNCF = UserKNNCFRecommender(URM_train)
userKNNCF.fit(topK=10, shrink=50, similarity="jaccard")

itemKNNCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
itemKNNCBF.fit(topK=10, shrink=50, similarity="jaccard")

pureSVD = PureSVDRecommender(URM_train)
pureSVD.fit()



for index in range(0, maxIndex):

    print("Cluster number: " + str(index))
    toIgnore = [x for x in users if x not in clusterSamples[index]]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=toIgnore)

    results, _ = evaluator_test.evaluateRecommender(itemKNNCF)
    MAP_itemKNNCF_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(userKNNCF)
    MAP_userKNNCF_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
    MAP_itemKNNCBF_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(pureSVD)
    MAP_pureSVD_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(elasticNet)
    MAP_elasticNet_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(sslim)
    MAP_sslim_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(topPop)
    MAP_topPop_per_group.append(results[cutoff]["MAP"])

pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
pyplot.plot(MAP_userKNNCF_per_group, label="userKNNCF")
pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
pyplot.plot(MAP_elasticNet_per_group, label="elasticNet")
pyplot.plot(MAP_sslim_per_group, label="sslim")
pyplot.plot(MAP_topPop_per_group, label="topPop")
pyplot.ylabel('MAP')
pyplot.xlabel('User Group')
pyplot.legend()
pyplot.savefig("userWise.jpg")