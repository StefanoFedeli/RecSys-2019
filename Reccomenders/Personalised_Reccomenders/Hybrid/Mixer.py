import evaluator as evaluate
import scipy.sparse as sps
import utils_new as utils
import random
import numpy as np

from External_Libraries.GraphBased.P3alphaRecommender import P3alphaRecommender
from External_Libraries.GraphBased.RP3betaRecommender import RP3betaRecommender
from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Reccomenders.Personalised_Reccomenders.Collaborative_Filtering.Slim.slimbpr import SLIM_BPR_Recommender
from Reccomenders.Personalised_Reccomenders.Hybrid.SSlim import ReccomenderSslim
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

class HybridReccomender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, Rec_4, Rec_5, Rec_6, Rec_7):
        super(HybridReccomender, self).__init__(URM_train)

        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.Recommender_4 = Rec_4
        self.Recommender_5 = Rec_5
        self.Recommender_6 = Rec_6
        self.Recommender_7 = Rec_7

        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0

    def __str__(self):
        print("A:{0:.3f}, B:{1:.3f}, C:{2:.3f}, D:{3:.3f}, E:{4:.3f}, F:{5:.3f}, G:{6:.3f}".format(
                                                                       self.alpha, self.beta, self.gamma,
                                                                       self.d, self.e, self.f, self.g))

    def fit(self, alpha=0., beta=0., gamma=0., d=0., e=0., f=0., g=0.):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.d = d
        self.e = e
        self.f = f
        self.g = g

    def _compute_item_score(self, user_id_array, items_to_compute=False):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.Recommender_4._compute_item_score(user_id_array)
        item_weights_5 = self.Recommender_5._compute_item_score(user_id_array)
        item_weights_6 = self.Recommender_6._compute_item_score(user_id_array)
        item_weights_7 = self.Recommender_7._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha +\
                       item_weights_2 * self.beta +\
                       item_weights_3 * self.gamma +\
                       item_weights_4 * self.d +\
                       item_weights_5 * self.e +\
                       item_weights_6 * self.f +\
                       item_weights_7 * self.g

        return item_weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, at=10,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        result = super().recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag,
                                   remove_custom_items_flag, return_scores)
        if return_scores is True:
            return result
        else:
            return result[:at]


URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
ICM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/ICM_all.npz"))
users = utils.get_target_users("../../../Dataset/target_users.csv", seek=8)
URM = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
validator = validate(URM_test, [10])


sslim = ReccomenderSslim(URM)
sslim.fit(train="")

CFItem = ItemKNNCFRecommender(URM)
CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

slim = SLIM_BPR_Recommender(URM)
slim.fit(path="../../../")

CFUser = UserKNNCFRecommender(URM)
CFUser.fit(703, 25, "asymmetric")

CBItem = ItemKNNCBFRecommender(URM,ICM_all)
CBItem.fit(shrink=10, topK=10, similarity="asymmetric", feature_weighting="none", normalize=True)

P3a = P3alphaRecommender(URM)
P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

P3b = RP3betaRecommender(URM)
P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

"""
rec_sys = HybridReccomender(URM, CFItem, CFUser, slim, CBItem, P3a, P3b, sslim)
rec_sys.fit(0.223, 0.166, 0.331, 0.725, 2.031, 2.524, 1.777)
"""

rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, CFUser)
rec_sys.fit(0.186, 1.812, 1.746, 1.744, 0, 0, 0)

"""
for i in range(120):
    SS = random.uniform(0.1, 2.3)
    Ga = random.uniform(0.1, 2.3)
    Gb = random.uniform(0.1, 2.3)
    CBI = random.uniform(0.1, min(SS,Ga,Gb))
    CFI = random.uniform(0.1, 2.3)
    CFU = random.uniform(0.1, 2.3)
    Sli = random.uniform(0.1, 2.3)
    SS = SS if bool(random.getrandbits(1)) is False else 0
    Ga = Ga if bool(random.getrandbits(1)) is False else 0
    Gb = Gb if bool(random.getrandbits(1)) is False else 0
    CBI = CBI if bool(random.getrandbits(1)) is False else 0
    CFI = CFI if bool(random.getrandbits(1)) is False else 0
    CFU = CFU if bool(random.getrandbits(1)) is False else 0
    Sli = Sli if bool(random.getrandbits(1)) is False else 0

    rec_sys.fit(CBI, Ga, Gb, SS, Sli, CFI, CFU)
    #print(rec_sys.__str__())
    #print(evaluate.evaluate(users, rec_sys, URM_test, 10)["MAP"])
    results = validator.evaluateRecommender(rec_sys)
    if results[0][10]["MAP"] > 0.0288:
        print("***** GOT IT ******* GOT IT")
        print(rec_sys.__str__())
        print(results[0][10]["MAP"])
        print("***** GOT IT ******* GOT IT")
    #print(results[0][10])
    print("\n")

"""
    #A:0.223, B:0.169, C:0.331, D:0.725, E:2.031, F:2.524, G:1.777 @ 0.027578921007649693
    #A:0.186, B:1.812, C:1.746, D:1.744, E:0.000, F:0.000, G:0.000 @ 0.02787492028548719

with open("../../../Outputs/HybridSte.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        # print(user_id)
        f.write(str(user_id) + "," + utils.trim(np.array(rec_sys.recommend(user_id))) + "\n")