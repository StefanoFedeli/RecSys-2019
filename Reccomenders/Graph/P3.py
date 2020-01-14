
import scipy.sparse as sps
import utils_new as utils
import evaluator as evaluate
import random

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.GraphBased.P3alphaRecommender import P3alphaRecommender
from External_Libraries.GraphBased.RP3betaRecommender import RP3betaRecommender



URM = sps.load_npz("../../Dataset/old/data_train.npz")
URM_test = sps.csr_matrix(sps.load_npz("../../Dataset/old/data_test.npz"))
users = utils.get_target_users("../../Dataset/target_users.csv", seek=8)
validator = validate(URM_test, [10])

recommender = P3alphaRecommender(URM)
recommender.fit(alpha=0.25662502344934046, min_rating=0, topK=25, implicit=True, normalize_similarity=True)
evaluate.evaluate(users, recommender, URM_test, 10)
'''
for i in range(15):
    alpha = random.uniform(0.2, 0.4)
    for k in [25]:
            for norm in [True]:
                recommender.fit(k,alpha,0,True,norm)
                print(recommender.__str__())
                print(evaluate.evaluate(users, recommender, URM_test, 10)["MAP"])
                results = validator.evaluateRecommender(recommender)
                print(results[0][10]["MAP"])
                print("\n")

# P3alpha(alpha=0.25662502344934046, min_rating=0, topk=25, implicit=True, normalize_similarity=True)
'''
print("***************************")
print("**  BETA START WORKING   **")
print("***************************")
recommenderB = RP3betaRecommender(URM)
recommenderB.fit(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topK=25, implicit=True, normalize_similarity=True)

print(evaluate.evaluate(users, recommenderB, URM_test, 10)["MAP"])
results = validator.evaluateRecommender(recommenderB)
print(results[0][10]["MAP"])
print("\n")

# RP3beta(alpha=0.22218786834129392, beta=0.23468317063424235, min_rating=0, topk=25, implicit=True, normalize_similarity=True) @ 0.0239
# RP3beta(alpha=0.23509260801291312, beta=0.1864977404134703, min_rating=0, topk=32, implicit=True, normalize_similarity=True)
# RP3beta(alpha=0.17690989624920614, beta=0.18014827953297588, min_rating=0, topk=32, implicit=True, normalize_similarity=True)