import utils_new as utils
import numpy as np
import scipy.sparse as sps
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

#utils.compare_csv("../../../../../Outputs/TopPop_freeze.csv", "../../../../../Outputs/TopPop_CBI_all.csv")
#exit()

features1 = utils.get_second_column("../../../../../Dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../../../Dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../../../Dataset/ICM_asset.csv")
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../../../Dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../../../Dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../../../Dataset/ICM_asset.csv")
items = items1 + items2 + items3

ones = np.ones(len(features))

URM_original = sps.load_npz("../../../../../Dataset/data_all.npz")

n_items = URM_original.shape[1]
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()


class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit(self, topK, shrink, similarity, normalize = True):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
        print(self.W_sparse.data)
        np.save("../../../../../Dataset/Similarity.npy", self.W_sparse)

    def recommend(self, user_id, at=None):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

for i in range(0, 10):
    URM = utils.getURMfromOUTPUT("../../../../../Outputs/TopPop_cold.csv", i,
                                 shape=(URM_original.shape[0], URM_original.shape[1])).tocsr()
    recommender = ItemCBFKNNRecommender(URM, ICM_all)
    recommender.fit(shrink=50, topK=10, similarity="jaccard")
    users = utils.get_target_users("../../../../../Dataset/target_users_cold.csv")
    with open("../../../../../Outputs/topPop_CBI_cold_" + str(i) + ".csv", 'w') as f:
        # f.write("user_id,item_list\n")
        for user_id in users:
            f.write(str(user_id) + "," + utils.trim(recommender.recommend(user_id, at=10)) + "\n")

outputs = utils.mergeFirstChoices("../../../../../Outputs/TopPop_CBI_cold_")
with open("../../../../../Outputs/TopPop_CBI_cold_all.csv", 'w') as f:
    for key in outputs.keys():
        f.write(str(key) + "," + utils.trim(outputs.get(key)) + "\n")