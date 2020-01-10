import utils_new as utils
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
import evaluator as evaluate
from External_Libraries.DataIO import DataIO
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate

'''
x_tick = [10, 20, 50, 100, 200]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=50, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("topk.png")

print("DONE")

x_tick = [10, 20, 50, 100, 200]
MAP_per_shrinkage = []

for shrink in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=10)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])


pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")

#users = utils.get_target_users("../../../../../Dataset/users_clusters/Coll_I.csv")
users = utils.get_target_users("../../../../../Dataset/target_users.csv")
with open("../../../../../Outputs/Coll_I.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(recommender.recommend(user_id, at=10)) + "\n")
'''
URM_test = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_test.npz"))
users = utils.get_target_users("../../../../../Dataset/target_users.csv", seek=9)
URM = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_train.npz"))
validator = validate(URM_test, [10])

mauri_recsys = ItemKNNCFRecommender(URM)
mauri_recsys.fit(shrink=106, topK=63, similarity="jaccard", feature_weighting="TF-IDF",normalize=False)
print(evaluate.evaluate(users, mauri_recsys, URM_test, 10)["MAP"])
results = validator.evaluateRecommender(mauri_recsys)
print(results[0][10]["MAP"])


# SHRINK:10, K:100, SIMILARITY:asymmetric, FEATURE=none, NORM=False @ 0.023746
