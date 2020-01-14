import pickle

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
from External_Libraries.MatrixFactorization.NMFRecommender import NMFRecommender
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Reccomenders.Personalised_Reccomenders.Hybrid.PureHybrid import HybridReccomender
from Reccomenders.Personalised_Reccomenders.Collaborative_Filtering.Slim.SLIM_ElasticNet import ElasticNetRecommender
import random

def create_clusters():
    user_list = list(range(0, 30911))
    users_dict = dict()

    cold = utils.get_target_users("../../../Dataset/target_users_cold.csv", seek=8)
    interaction_age, user_age, age = utils.create_tuples("../../../Dataset/UCM_age.csv",13)
    interaction_region, user_region, region = utils.create_tuples("../../../Dataset/UCM_region.csv",13)

    for u in user_list:
        if u in cold:
            user_list.remove(u)
            continue
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
    users_dict[10000] = list()
    users_dict[10001] = cold

    print(users_dict)

    clusters = []

    for key in users_dict.copy().keys():
        if len(users_dict[key]) < 600:
            for u in users_dict.get(key):
                users_dict[10000].append(u)
            users_dict.pop(key)

    for key in users_dict:
        clusters.append(users_dict[key])

    return clusters




URM = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))

ICM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/ICM_all.npz"))

clusterSamples = create_clusters()
maxIndex = len(clusterSamples)
users = range(0, 30911)

MAP_pureSVD_per_group = []
MAP_sslim_per_group = []
MAP_CFItem_per_group = []
MAP_slim_per_group = []
MAP_CBItem_per_group = []
MAP_P3a_per_group = []
MAP_P3b_per_group = []
MAP_Hybrid_per_group = []
cutoff = 10

pureSVD = PureSVDRecommender(URM)
pureSVD.fit(526, True)

sslim = ReccomenderSslim(URM)
sslim.fit(train="")

CFItem = ItemKNNCFRecommender(URM)
CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

slim = SLIM_BPR_Recommender(URM)
slim.fit(path="../../../", train="")

#CFUser = UserKNNCFRecommender(URM)
#CFUser.fit(703, 25, "asymmetric")

CBItem = ItemKNNCBFRecommender(URM,ICM_all)
CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)

P3a = P3alphaRecommender(URM)
P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

P3b = RP3betaRecommender(URM)
P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

elasticNet = ElasticNetRecommender(URM)
elasticNet.fit("")

rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, pureSVD, elasticNet, pureSVD, pureSVD)
rec_sys.fit(0, 1.866, 0.589, 0.421, 1.219, 0, 0, 1.254, 1.415, 0)

rec_list = (pureSVD,sslim,CFItem,slim,CBItem,P3a,P3b,rec_sys)

best_ones = dict()

def run():
    best_names = dict()

    for index in range(0, maxIndex):

        best_ones[index] = None
        print("Cluster number: " + str(index))
        toIgnore = [x for x in users if x not in clusterSamples[index]]
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=toIgnore)

        results, _ = evaluator_test.evaluateRecommender(pureSVD)
        MAP_pureSVD_per_group.append(results[cutoff]["MAP"])
        max = MAP_pureSVD_per_group[index]
        best_ones[index] = pureSVD
        best_names[index] = pureSVD.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(sslim)
        MAP_sslim_per_group.append(results[cutoff]["MAP"])
        if MAP_sslim_per_group[index] > max:
            max = MAP_sslim_per_group[index]
            best_ones[index] = sslim
            best_names[index] = sslim.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(CFItem)
        MAP_CFItem_per_group.append(results[cutoff]["MAP"])
        if MAP_CFItem_per_group[index] > max:
            max = MAP_CFItem_per_group[index]
            best_ones[index] = CFItem
            best_names[index] = CFItem.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(slim)
        MAP_slim_per_group.append(results[cutoff]["MAP"])
        if MAP_slim_per_group[index] > max:
            max = MAP_slim_per_group[index]
            best_ones[index] = slim
            best_names[index] = slim.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(CBItem)
        MAP_CBItem_per_group.append(results[cutoff]["MAP"])
        if MAP_CBItem_per_group[index] > max:
            max = MAP_CBItem_per_group[index]
            best_ones[index] = CBItem
            best_names[index] = CBItem.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(P3a)
        MAP_P3a_per_group.append(results[cutoff]["MAP"])
        if MAP_P3a_per_group[index] > max:
            max = MAP_P3a_per_group[index]
            best_ones[index] = P3a
            best_names[index] = P3a.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(P3b)
        MAP_P3b_per_group.append(results[cutoff]["MAP"])
        if MAP_P3b_per_group[index] > max:
            max = MAP_P3b_per_group[index]
            best_ones[index] = P3b
            best_names[index] = P3b.RECOMMENDER_NAME

        results, _ = evaluator_test.evaluateRecommender(rec_sys)
        MAP_Hybrid_per_group.append(results[cutoff]["MAP"])
        if MAP_Hybrid_per_group[index] > max:
            max = MAP_Hybrid_per_group[index]
            best_ones[index] = rec_sys
            best_names[index] = rec_sys.RECOMMENDER_NAME

        print("The best is "+ best_names[index] +" with: "+ str(max) +"("+ str(URM.shape[0]-len(toIgnore)) +" users)")

    with open('../../../Dataset/userWise.p', 'wb') as fp:
        pickle.dump(best_names, fp, protocol=pickle.HIGHEST_PROTOCOL)

#run()
#exit()

with open('../../../Dataset/userWise.p', 'rb') as fp:
    best_names = dict(pickle.load(fp))
    for id in best_names.keys():
        for rec in rec_list:
            if rec.RECOMMENDER_NAME == best_names.get(id):
                best_ones[id] = rec
print(best_ones)


users = utils.get_target_users("../../../Dataset/target_users.csv", seek=8)
count = 0
with open("../../../Outputs/UserWise.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for u in users:
        if u % 7500 == 0:
            print("user {0} of {1}".format(u,len(users)))
        rec = list()
        for index in range(0,len(clusterSamples)):
            if u in clusterSamples[index]:
                rec.append(best_ones[index].recommend(u, cutoff=10))
        if len(rec) > 1:
            score = dict()
            for i in range(0,len(rec)):
                for j in rec[i]:
                    if j in score:
                        score[j] += 1
                    else:
                        score[j] = 1
            max = 0
            #print(score)
            rec = list()
            rec.append(list())
            for w in sorted(score, key=score.get, reverse=True):
                if max > 9:
                    break
                rec[0].append(w)
                max += 1
            count += 1
            #print("user {0} in more than 1 cluster".format(u))
        #print(rec)
        f.write(str(u) + "," + utils.trim(np.array(rec[0])) + "\n")

print(count)
"""
pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
pyplot.plot(MAP_sslim_per_group, label="sslim")
pyplot.plot(MAP_CFItem_per_group, label="CFItem")
pyplot.plot(MAP_slim_per_group, label="slim")
pyplot.plot(MAP_CFUser_per_group, label="CFUser")
pyplot.plot(MAP_CBItem_per_group, label="CBItem")
pyplot.plot(MAP_P3a_per_group, label="P3a")
pyplot.plot(MAP_P3b_per_group, label="P3b")
pyplot.plot(MAP_NMF_per_group, label="elasticNet")

pyplot.ylabel('MAP')
pyplot.xlabel('User Group')
pyplot.legend()
pyplot.savefig("userWise.png")
"""