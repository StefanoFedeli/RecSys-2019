import utils_new as utils
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as pyplot

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout
from External_Libraries.GraphBased.P3alphaRecommender import P3alphaRecommender
from External_Libraries.GraphBased.RP3betaRecommender import RP3betaRecommender
from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Reccomenders.Personalised_Reccomenders.Collaborative_Filtering.Slim.slimbpr import SLIM_BPR_Recommender
from Reccomenders.Personalised_Reccomenders.Hybrid.SSlim import ReccomenderSslim
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Reccomenders.Personalised_Reccomenders.Collaborative_Filtering.Slim.SLIM_ElasticNet import ElasticNetRecommender
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

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
        clusterSamples.append(users_dict[key])

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

MAP_pureSVD_per_group = []
MAP_sslim_per_group = []
MAP_CFItem_per_group = []
MAP_slim_per_group = []
MAP_CFUser_per_group = []
MAP_CBItem_per_group = []
MAP_P3a_per_group = []
MAP_P3b_per_group = []
MAP_elasticNet_per_group = []
cutoff = 10

elasticNet = ElasticNetRecommender(URM_train)
elasticNet.fit("train")

pureSVD = PureSVDRecommender(URM_train)
pureSVD.fit()

sslim = ReccomenderSslim(URM_train)
sslim.fit(train="-train")

CFItem = ItemKNNCFRecommender(URM_train)
CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

slim = SLIM_BPR_Recommender(URM_train)
slim.fit(path="../../../")

CFUser = UserKNNCFRecommender(URM_train)
CFUser.fit(703, 25, "asymmetric")

CBItem = ItemKNNCBFRecommender(URM_train,ICM_all)
CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)

P3a = P3alphaRecommender(URM_train)
P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

P3b = RP3betaRecommender(URM_train)
P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)



for index in range(0, maxIndex):

    max = 0
    name = "none"

    print("Cluster number: " + str(index))
    toIgnore = [x for x in users if x not in clusterSamples[index]]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=toIgnore)

    results, _ = evaluator_test.evaluateRecommender(pureSVD)
    MAP_pureSVD_per_group.append(results[cutoff]["MAP"])
    max = MAP_pureSVD_per_group[index]
    name = "pureSVD"

    results, _ = evaluator_test.evaluateRecommender(sslim)
    MAP_sslim_per_group.append(results[cutoff]["MAP"])
    if MAP_sslim_per_group[index] > max:
        max = MAP_sslim_per_group[index]
        name = "sslim"

    results, _ = evaluator_test.evaluateRecommender(CFItem)
    MAP_CFItem_per_group.append(results[cutoff]["MAP"])
    if MAP_CFItem_per_group[index] > max:
        max = MAP_CFItem_per_group[index]
        name = "CFItem"

    results, _ = evaluator_test.evaluateRecommender(slim)
    MAP_slim_per_group.append(results[cutoff]["MAP"])
    if MAP_slim_per_group[index] > max:
        max = MAP_slim_per_group[index]
        name = "slim"

    results, _ = evaluator_test.evaluateRecommender(CFUser)
    MAP_CFUser_per_group.append(results[cutoff]["MAP"])
    if MAP_CFUser_per_group[index] > max:
        max = MAP_CFUser_per_group[index]
        name = "CFUser"

    results, _ = evaluator_test.evaluateRecommender(CBItem)
    MAP_CBItem_per_group.append(results[cutoff]["MAP"])
    if MAP_CBItem_per_group[index] > max:
        max = MAP_CBItem_per_group[index]
        name = "CBItem"

    results, _ = evaluator_test.evaluateRecommender(P3a)
    MAP_P3a_per_group.append(results[cutoff]["MAP"])
    if MAP_P3a_per_group[index] > max:
        max = MAP_P3a_per_group[index]
        name = "P3a"

    results, _ = evaluator_test.evaluateRecommender(P3b)
    MAP_P3b_per_group.append(results[cutoff]["MAP"])
    if MAP_P3b_per_group[index] > max:
        max = MAP_P3b_per_group[index]
        name = "P3b"

    results, _ = evaluator_test.evaluateRecommender(elasticNet)
    MAP_elasticNet_per_group.append(results[cutoff]["MAP"])
    if MAP_elasticNet_per_group[index] > max:
        max = MAP_elasticNet_per_group[index]
        name = "elasticNet"

    print("The best is " + name + " with: " + str(max))



pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
pyplot.plot(MAP_sslim_per_group, label="sslim")
pyplot.plot(MAP_CFItem_per_group, label="CFItem")
pyplot.plot(MAP_slim_per_group, label="slim")
pyplot.plot(MAP_CFUser_per_group, label="CFUser")
pyplot.plot(MAP_CBItem_per_group, label="CBItem")
pyplot.plot(MAP_P3a_per_group, label="P3a")
pyplot.plot(MAP_P3b_per_group, label="P3b")
pyplot.plot(MAP_elasticNet_per_group, label="elasticNet")

pyplot.ylabel('MAP')
pyplot.xlabel('User Group')
pyplot.legend()
pyplot.savefig("userWise.png")