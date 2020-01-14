import utils_new as utils
import random
import scipy.sparse as sps
import evaluator as evaluate

from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender as reccomender
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate

'''
x_tick = [500, 600, 750, 1000, 1200]
MAP_per_k = []

for topK in x_tick:
    recommender = UserCFKNNRecommender(URM_train)
    recommender.fit(shrink=0.0, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("topk.png")


x_tick = [0, 10, 50, 100, 200, 500]
MAP_per_shrinkage = []

for shrink in x_tick:
    recommender = UserCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=600)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])


pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")

similary = ["cosine", "asymmetric", "jaccard", "dice", "tversky", "tanimoto"]
#for sim in range(3):
for sim in range(5,2,-1):
    print("++++++ WORKING ON {0} +++++++".format(similary[sim]))
    for i in range(10):
        k = random.randint(700, 800)
        s = random.randint(24, 28)
        ft = "none"
        for norm in [True]:
            print("SHRINK:{0}, K:{1}, SIMILARITY:{2}, FEATURE={3}, NORM={4}".format(s,k,similary[sim],ft,norm))
            mauri_recsys.fit(k,s,similary[sim],norm,ft)
            print(evaluate.evaluate(users, mauri_recsys, URM_test, 10)["MAP"])
            results = validator.evaluateRecommender(mauri_recsys)
            print(results[0][10]["MAP"])
            print("\n")

'''

URM_test = sps.csr_matrix(sps.load_npz("../../../../Dataset/old/data_test.npz"))
users = utils.get_target_users("../../../../Dataset/target_users.csv", seek=8)
URM = sps.csr_matrix(sps.load_npz("../../../../Dataset/old/data_train.npz"))

validator = validate(URM_test, [10])
mauri_recsys = reccomender(URM)
mauri_recsys.fit(703, 25, "asymmetric")

print(evaluate.evaluate(users, mauri_recsys, URM_test, 10)["MAP"])
results = validator.evaluateRecommender(mauri_recsys)
print(results[0][10]["MAP"])
print("\n")

# SHRINK:27, K:748, SIMILARITY:cosine, FEATURE=none, NORM=True @ 0.017748
# SHRINK:24, K:711, SIMILARITY:tanimoto, FEATURE=none, NORM=True @ 0.017710
# SHRINK:25, K:703, SIMILARITY:asymmetric, FEATURE=none, NORM=True @ 0.017792