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
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Reccomenders.Personalised_Reccomenders.Collaborative_Filtering.Slim.SLIM_ElasticNet import ElasticNetRecommender
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from External_Libraries.MatrixFactorization.NMFRecommender import NMFRecommender
from Reccomenders.Personalised_Reccomenders.Hybrid.PureHybrid import HybridReccomender

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout

URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
ICM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/ICM_all.npz"))
users = utils.get_target_users("../../../Dataset/target_users.csv", seek=9)
URM = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
validator = validate(URM_test, [10])

CBItem = ItemKNNCBFRecommender(URM,ICM_all)
CBItem.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)

CFItem = ItemKNNCFRecommender(URM)
CFItem.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

CFUser = UserKNNCFRecommender(URM)
CFUser.fit(703, 25, "asymmetric")

P3a = P3alphaRecommender(URM)
P3a.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

P3b = RP3betaRecommender(URM)
P3b.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

elasticNet = ElasticNetRecommender(URM)
elasticNet.fit("train")


sslim = ReccomenderSslim(URM)
sslim.fit(train="-train")

slim = SLIM_BPR_Recommender(URM)
slim.fit(path="../../../", train="-train")

pureSVD = PureSVDRecommender(URM)
pureSVD.fit(526, True)

NMF = NMFRecommender(URM)
NMF.fit(360, 0.48, "multiplicative_update", "nndsvda", "frobenius", False, None)

"""
rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, CFUser)
rec_sys.fit(0.186, 1.812, 1.746, 1.744, 0, 0, 0)
"""

rec_sys = HybridReccomender(URM, CBItem, P3a, P3b, sslim, slim, CFItem, CFUser, elasticNet, pureSVD, NMF)
target_users = utils.get_target_users("../../../Dataset/target_users.csv")

for i in range(200):

    CBI_param = random.uniform(0, 0.5)
    CFI_param = random.uniform(0.5, 1)
    CFU_param = random.uniform(0, 0.5)
    P3a_param = random.uniform(0.5, 1)
    P3b_param = random.uniform(0.5, 1)
    elasticNet_param = random.uniform(0.5, 1)

    rec_sys.fit(CBI_param, P3a_param, P3b_param, 0, 0, CFI_param, CFU_param, elasticNet_param, 0, 0)
    # print(rec_sys.__str__())
    # print(evaluate.evaluate(users, rec_sys, URM_test, 10)["MAP"])
    results = validator.evaluateRecommender(rec_sys)
    # print(results)
    if results[0][10]["MAP"] > 0.0296:
        print("\n***** GOT IT ******* GOT IT")
        print(rec_sys.__str__())
        print(results[0][10]["MAP"])
        print("***** GOT IT ******* GOT IT\n")
    else:
        print("{0}: {1} @ {2:.5f}".format(i, rec_sys, results[0][10]["MAP"]))

#A:0.186, B:1.812, C:1.746, D:1.744, E:0.000, F:0.000, G:0.000 @ 0.02787492028548719

    #   CBI,    Ga,       Gb,      SS,     Sli,    CFI,      CFU,    Ela,     svd,      nmf
    #A:0.000, B:1.866, C:0.589, D:0.421, E:1.219, F:0.000, G:0.000, H:1.254 I:1.415, L:0.000 @ 0.029694638816818258

"""
rec_sys.fit(0, 1.866, 0.589, 0.421, 1.219, 0, 0, 1.254, 1.415, 0)
with open("../../../Outputs/HybridSteNew.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        # print(user_id)
        f.write(str(user_id) + "," + utils.trim(np.array(rec_sys.recommend(user_id))) + "\n")
"""