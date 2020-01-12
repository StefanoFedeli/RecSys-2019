import scipy.sparse as sps
import utils_new as utils
import evaluator as evaluate
import random

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.MatrixFactorization.NMFRecommender import NMFRecommender

URM = sps.load_npz("../../../Dataset/data_train.npz")
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
users = utils.get_target_users("../../../Dataset/target_users.csv", seek=8)
validator = validate(URM_test, [10])

recommender = NMFRecommender(URM)
recommender.fit(360, 0.48, "multiplicative_update", "nndsvda", "frobenius", False, None)
results = validator.evaluateRecommender(recommender)
print(results[0][10]["MAP"])
"""
for fac in range(350,560,20):
    for sol in ["coordinate_descent", "multiplicative_update"]:
        for b in ["frobenius"]:
            for init in ["nndsvda", "random"]:
                for seed in [False, True, None]:
                    l1 = random.uniform(0.1, 0.9)
                    recommender.fit(fac,l1,sol,init,b,False,seed)
                    print("FACTOR:{0}, SEED:{1}, L1:{2}, BETA:{3}, INIT:{4}, SOLVER:{5}".format(
                        fac, seed, l1, b, init, sol))
                    print(evaluate.evaluate(users, recommender, URM_test, 10)["MAP"])
                    #results = validator.evaluateRecommender(recommender)
                    #print(results[0][10]["MAP"])
                    print("\n")

"""
#FACTOR:360, SEED:False, L1:0.48, BETA:frobenius, INIT:nndsvda, SOLVER:multiplicative_update @ 0.0151