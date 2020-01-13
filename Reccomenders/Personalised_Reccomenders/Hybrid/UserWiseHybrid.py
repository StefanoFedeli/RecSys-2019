import utils_new as utils
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
import evaluator
import pickle

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
from External_Libraries.Base.NonPersonalizedRecommender import TopPop

def create_clusters():
    user_list = list(range(0, 30911))
    users_dict = dict()

    interaction_age, user_age, age = utils.create_tuples("../../../Dataset/UCM_age.csv", 14)
    interaction_region, user_region, region = utils.create_tuples("../../../Dataset/UCM_region.csv", 14)

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

        if target_age >= 0 and target_region >= 0:
            key = int(str(target_age) + str(target_region))
            if key not in users_dict:
                users_dict[key] = list()
            users_dict[key].append(u)
            user_list.remove(u)
        elif target_age < 0 and target_region >= 0:
            for i in range(0, max(age)):
                key = int(str(i) + str(target_region))
                if key not in users_dict:
                    users_dict[key] = list()
                users_dict[key].append(u)
            user_list.remove(u)
        elif target_age >= 0 and target_region < 0:
            for i in range(min(region), max(region)):
                key = int(str(target_age) + str(i))
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

class recommenderEntry:
    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name

def recommender_setup_train():
    recommenders = []

    topPop = TopPop(URM_train)
    topPop.fit()
    topPopRecommender = recommenderEntry(topPop, "topPop")
    recommenders.append(topPopRecommender)

    elasticNet = ElasticNetRecommender(URM_train)
    elasticNet.fit("train")
    elasticNetRecommender = recommenderEntry(elasticNet, "elasticNet")
    recommenders.append(elasticNetRecommender)

    pureSVD = PureSVDRecommender(URM_train)
    pureSVD.fit()
    pureSVDRecommender = recommenderEntry(pureSVD, "pureSVD")
    recommenders.append(pureSVDRecommender)

    sslim = ReccomenderSslim(URM_train)
    sslim.fit(train="-train")
    sslimRecommender = recommenderEntry(sslim, "sslim")
    recommenders.append(sslimRecommender)

    CFItem = ItemKNNCFRecommender(URM_train)
    CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)
    CFItemRecommender = recommenderEntry(CFItem, "CFItem")
    recommenders.append(CFItemRecommender)

    slim = SLIM_BPR_Recommender(URM_train)
    slim.fit(path="../../../", train="-train")
    slimRecommender = recommenderEntry(slim, "slim")
    recommenders.append(slimRecommender)

    CFUser = UserKNNCFRecommender(URM_train)
    CFUser.fit(703, 25, "asymmetric")
    CFUserRecommender = recommenderEntry(CFUser, "CFUser")
    recommenders.append(CFUserRecommender)

    CBItem = ItemKNNCBFRecommender(URM_train, ICM_all)
    CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)
    CBItemRecommender = recommenderEntry(CBItem, "CBItem")
    recommenders.append(CBItemRecommender)

    P3a = P3alphaRecommender(URM_train)
    P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)
    P3aRecommender = recommenderEntry(P3a, "P3a")
    recommenders.append(P3aRecommender)

    P3b = RP3betaRecommender(URM_train)
    P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True,
            normalize_similarity=True)
    P3bRecommender = recommenderEntry(P3b, "P3b")
    recommenders.append(P3bRecommender)

    return recommenders

def recommender_setup_all():
    recommenders = []

    topPop = TopPop(URM_all)
    topPop.fit()
    topPopRecommender = recommenderEntry(topPop, "topPop")
    recommenders.append(topPopRecommender)

    elasticNet = ElasticNetRecommender(URM_all)
    elasticNet.fit("all")
    elasticNetRecommender = recommenderEntry(elasticNet, "elasticNet")
    recommenders.append(elasticNetRecommender)

    pureSVD = PureSVDRecommender(URM_all)
    pureSVD.fit()
    pureSVDRecommender = recommenderEntry(pureSVD, "pureSVD")
    recommenders.append(pureSVDRecommender)

    sslim = ReccomenderSslim(URM_all)
    sslim.fit()
    sslimRecommender = recommenderEntry(sslim, "sslim")
    recommenders.append(sslimRecommender)

    CFItem = ItemKNNCFRecommender(URM_all)
    CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)
    CFItemRecommender = recommenderEntry(CFItem, "CFItem")
    recommenders.append(CFItemRecommender)

    slim = SLIM_BPR_Recommender(URM_all)
    slim.fit(path="../../../")
    slimRecommender = recommenderEntry(slim, "slim")
    recommenders.append(slimRecommender)

    CFUser = UserKNNCFRecommender(URM_all)
    CFUser.fit(703, 25, "asymmetric")
    CFUserRecommender = recommenderEntry(CFUser, "CFUser")
    recommenders.append(CFUserRecommender)

    CBItem = ItemKNNCBFRecommender(URM_all, ICM_all)
    CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)
    CBItemRecommender = recommenderEntry(CBItem, "CBItem")
    recommenders.append(CBItemRecommender)

    P3a = P3alphaRecommender(URM_all)
    P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)
    P3aRecommender = recommenderEntry(P3a, "P3a")
    recommenders.append(P3aRecommender)

    P3b = RP3betaRecommender(URM_all)
    P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True,
            normalize_similarity=True)
    P3bRecommender = recommenderEntry(P3b, "P3b")
    recommenders.append(P3bRecommender)

    return recommenders

def getRecommender(name, recommenders):
    for item in recommenders:
        if item.name == name:
            return item.recommender
    return None

class clusterElement:
    def __init__(self, userId):
        self.userId = userId
        self.score = 0
        self.recommender = None
    def checkAndUpdate(self, newRecommender, newScore):
        if newScore > self.score:
            self.recommender = newRecommender
            self.score = newScore

URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
ICM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/ICM_all.npz"))

clusterElements = []


recommendersTrain = recommender_setup_train()
clusterSamples = create_clusters()
maxIndex = len(clusterSamples)
users = range(0, 30911)

for index in range(0, 30911):
    clusterElements.append(clusterElement(index))

for index in range(0, maxIndex):
    maxMAP = 0
    bestRecommender = None
    print("Currently processing cluster number: " + str(index))
    for entry in recommendersTrain:
        print("Currently evaluating recommender: " + entry.name)
        score = evaluator.evaluate(clusterSamples[index], entry.recommender, URM_test, 10)
        if score.get("MAP") > maxMAP:
            maxMAP = score.get("MAP")
            bestRecommender = entry
    for user in clusterSamples[index]:
        clusterElements[user].checkAndUpdate(bestRecommender, maxMAP)
'''
with open('recommenderAssignment.pkl', 'wb') as output:
    pickle.dump(clusterElements, output, pickle.HIGHEST_PROTOCOL)


recommendersAll = recommender_setup_all()
targetUsers = utils.get_target_users("../../../Dataset/target_users.csv")

with open('recommenderAssignment.pkl', 'rb') as input:
    clusterElements = pickle.load(input)

with open("userWise.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user in targetUsers:
        recommender = getRecommender(clusterElements[user].recommender.name, recommendersAll)
        recommendations = recommender.recommend(user)
        f.write(str(user) + ", " + utils.trim(recommendations[:10]) + "\n")
'''