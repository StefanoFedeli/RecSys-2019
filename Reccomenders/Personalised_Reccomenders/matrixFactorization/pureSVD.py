import scipy.sparse as sps
import utils_new as utils
import evaluator as evaluate
import random

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

URM = sps.load_npz("../../../Dataset/data_train.npz")
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
users = utils.get_target_users("../../../Dataset/target_users.csv", seek=8)
validator = validate(URM_test, [10])

recommender = PureSVDRecommender(URM)

#for i in range(520, 541, 2):
#    for norm in [True, False]:
recommender.fit(526, True)
#print("EPOCHS:{0}, SEED:{1}".format(i, norm))
print(evaluate.evaluate(users, recommender, URM_test, 10)["MAP"])
results = validator.evaluateRecommender(recommender)
print(results[0][10]["MAP"])
print("\n")

# EPOCHS:526, SEED:True @ 0.015119