import utils_new as utils
import numpy as np
import scipy.sparse as sps
import evaluator as evaluate

from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender as reccomender
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate

features1 = utils.get_second_column("../../../../../Dataset/ICM_sub_class.csv",seek=13)
features2 = utils.get_second_column("../../../../../Dataset/ICM_price.csv",seek=13)
features3 = utils.get_second_column("../../../../../Dataset/ICM_asset.csv",seek=13)
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../../../Dataset/ICM_sub_class.csv",seek=13)
items2 = utils.get_first_column("../../../../../Dataset/ICM_price.csv",seek=13)
items3 = utils.get_first_column("../../../../../Dataset/ICM_asset.csv",seek=13)
items = items1 + items2 + items3


URM = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_all.npz"))

n_items = URM.shape[1]
n_tags = max(features) + 1

itemPopularity = URM.sum(axis=0)
itemPopularity = np.array(itemPopularity).squeeze()
popularItems = np.argsort(itemPopularity)[:int(n_items*0.1)]
coldItem = np.argsort(itemPopularity[itemPopularity == 0])


for i in range(len(popularItems)):
    items.append(popularItems[i])
    features.append(n_tags+1)
for i in range(len(coldItem)):
    items.append(coldItem[i])
    features.append(n_tags+2)

ones = np.ones(len(features))
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()

'''
x_tick = [10, 50, 100, 200, 500]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
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
    recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
    recommender.fit(shrink=shrink, topK=10)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")

'''
sps.save_npz("../../../../../Dataset/ICM_all.npz",ICM_all)

URM_test = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_test.npz"))
users = utils.get_target_users("../../../../../Dataset/target_users.csv", seek=8)
validator = validate(URM_test, [10])


mauri_recsys = reccomender(URM, ICM_all)
mauri_recsys.fit(shrink=10, topK=10, similarity="asymmetric", feature_weighting="none", normalize=True)
print(evaluate.evaluate(users, mauri_recsys, URM_test, 10)["MAP"])
results = validator.evaluateRecommender(mauri_recsys)
print(results[0][10]["MAP"])
print("\n")

# SHRINK:10, K:10, SIMILARITY:asymmetric, FEATURE=none, NORM=True @ 0.0088
# SHRINK:25, K:10, SIMILARITY:cosine, FEATURE=none, NORM=True @ 0.0088
