import scipy.sparse as sps
import utils_new as utils
import random
import numpy as np

from External_Libraries.GraphBased.P3alphaRecommender import P3alphaRecommender
from External_Libraries.GraphBased.RP3betaRecommender import RP3betaRecommender
from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Reccomenders.Collaborative_Filtering.Slim.slimbpr import SLIM_BPR_Recommender
from Reccomenders.Hybrid.SSlim import ReccomenderSslim
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Reccomenders.Collaborative_Filtering.Slim.SLIM_ElasticNet import ElasticNetRecommender
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from External_Libraries.MatrixFactorization.NMFRecommender import NMFRecommender
from Reccomenders.Hybrid.PureHybrid import HybridReccomender


URM_test = sps.csr_matrix(sps.load_npz("../../Dataset/URM/data_test.npz"))
ICM_all = sps.csr_matrix(sps.load_npz("../../Dataset/ICM/ICM_all.npz"))
users = utils.get_target_users("../../Dataset/target_users.csv", seek=9)
URM = sps.csr_matrix(sps.load_npz("../../Dataset/URM/data_all.npz"))
validator = validate(URM_test, [10])


sslim = ReccomenderSslim(URM)
sslim.fit(train="")

CFItem = ItemKNNCFRecommender(URM)
CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

slim = SLIM_BPR_Recommender(URM)
slim.fit(path="../../", train="")

CFUser = UserKNNCFRecommender(URM)
CFUser.fit(703, 25, "asymmetric")

CBItem = ItemKNNCBFRecommender(URM,ICM_all)
CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)

P3a = P3alphaRecommender(URM)
P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

P3b = RP3betaRecommender(URM)
P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

elasticNet = ElasticNetRecommender(URM)
elasticNet.fit(train="")

pureSVD = PureSVDRecommender(URM)
pureSVD.fit(526, True)

NMF = NMFRecommender(URM)
NMF.fit(360, 0.48, "multiplicative_update", "nndsvda", "frobenius", False, None)

"""
rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, CFUser)
rec_sys.fit(0.186, 1.812, 1.746, 1.744, 0, 0, 0)
"""

rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, CFUser, elasticNet, pureSVD, NMF)

"""
massim = 0.04127110796620984 #alpha=0.000, beta=9.055, gamma=12.351, d=2.284, e=0.000, f=10.030, g=1.186, h=8.928 i=0.000, l=1.500, norm:True
for i in range(35):

    SS = random.uniform(5, 15)
    Ga = random.uniform(0, 2)
    Gb = random.uniform(5, 15)
    CFI = random.uniform(1, 6)
    mas = max(CFI, Ga, Gb, SS) + 2
    Ela = random.uniform(5, 15)
    CBI = random.uniform(1, 3)
    CFU = random.uniform(0, 2)
    Sli = random.uniform(0, 2)
    svd = random.uniform(0, 2)
    nmf = random.uniform(1, 3)
    
    Ga = Ga if bool(random.getrandbits(1)) is False else 0
    CBI = CBI if bool(random.getrandbits(1)) is False else 0
    CFU = CFU if bool(random.getrandbits(1)) is False else 0
    Sli = Sli if bool(random.getrandbits(1)) is False else 0
    svd = svd if bool(random.getrandbits(1)) is False else 0
    nmf = nmf if bool(random.getrandbits(1)) is False else 0
    
    norm = bool(random.getrandbits(1))

    rec_sys.fit(norm, CBI, Ga, Gb, SS, Sli, CFI, CFU, Ela, svd, nmf)
    #print(rec_sys.__str__())
    #print(evaluate.evaluate(users, rec_sys, URM_test, 10)["MAP"])
    results = validator.evaluateRecommender(rec_sys)
    #print(results)
    if results[0][10]["MAP"] > massim:
        print("\n***** GOT IT ******* GOT IT")
        print(rec_sys.__str__())
        print(results[0][10]["MAP"])
        print("***** GOT IT ******* GOT IT\n")
        massim = results[0][10]["MAP"]
    else:
        print("{0}: {1} @ {2:.5f}".format(i, rec_sys, results[0][10]["MAP"]))
"""

    #   CBI,               Ga,          Gb,           SS,       Sli,        CFI,       CFU,         Ela,        svd,      nmf
    #alpha = 2.114, beta = 0.000, gamma = 12.227, d = 10.078, e = 0.000, f = 4.840, g = 0.000, h = 11.638, i = 0.000, l = 2.518, norm:True @ 0.041586937594092314

    #A:0.186, B:1.812, C:1.746, D:1.744, E:0.000, F:0.000, G:0.000 @ 0.02787492028548719
    #A:0.000, B:1.866, C:0.589, D:0.421, E:1.219, F:0.000, G:0.000, H:1.254 I:1.415, L:0.000 @ 0.029694638816818258
    #alpha=5.879, beta=11.393, gamma=12.836, d=1.600, e=0.000, f=13.970, g=0.000, h=11.623 i=0.000, l=0.000, norm:True @0.041352336690806776


rec_sys.fit(alpha=2.486, beta=1.606, gamma=10.264, d=7.918, e=1.713, f=5.300, g=1.301, h=11.257, i=0.331, l=2.299, n=True)


with open("../../Outputs/hybrid_norm_final2.csv", 'w') as f:
    f.write("user_id,item_list\n")

    user_batch_start = 0
    while user_batch_start < len(users):
        user_batch_end = user_batch_start + 1000
        user_batch_end = min(user_batch_end, len(users))

        test_user_batch_array = np.array(users[user_batch_start:user_batch_end])
        user_batch_start = user_batch_end

        # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
        recommended_items_batch_list, scores_batch = rec_sys.recommend(test_user_batch_array,
                                                                       remove_seen_flag=True,
                                                                       cutoff=10,
                                                                       remove_top_pop_flag=False,
                                                                       remove_custom_items_flag=False,
                                                                       return_scores=True)
        i=0
        for user_id in test_user_batch_array:
            # print(user_id)
            f.write(str(user_id) + "," + utils.trim(np.array(recommended_items_batch_list[i])) + "\n")
            i=i+1



