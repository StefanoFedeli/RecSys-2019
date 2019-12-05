import utils_new as utils
import numpy as np
from External_Libraries.Zeus.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from External_Libraries.Zeus.Compute_Similarity_Python import Compute_Similarity_Python

URM_all = utils.create_coo("../../../../../Dataset/URM.csv")
URM_all = URM_all.tocsr()

warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
warm_items = np.arange(URM_all.shape[1])[warm_items_mask]
#URM_all = URM_all[:, warm_items]

warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
warm_users = np.arange(URM_all.shape[0])[warm_users_mask]
warm_users = list(warm_users)
#URM_all = URM_all[warm_users, :]

URM_train, URM_test = split_train_leave_k_out_user_wise(URM_all, 1, False)


class ItemCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="jaccard"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

class TopPopRecommender(object):

    def fit(self, URM_train):

        self.URM_train = URM_train

        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=5, remove_seen=True):

        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items


'''
x_tick = [10, 50, 100, 200, 500]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=0, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender, URM_test.shape[0])
    MAP_per_k.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("topk.png")


x_tick = [0, 10, 50, 100, 200, 500]
MAP_per_shrinkage = []

for shrink in x_tick:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=100)

    result_dict = evaluate_algorithm(URM_test, recommender, URM_test.shape[0])
    MAP_per_shrinkage.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrink')
pyplot.savefig("shrink.png")
'''

recommender = ItemCFKNNRecommender(URM_train)
recommender.fit(shrink=50, topK=1000)
topPopRecommender_removeSeen = TopPopRecommender()
topPopRecommender_removeSeen.fit(URM_train)
users = utils.get_target_users("../../../../../Dataset/target_users.csv")
with open("output.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        if user_id not in warm_users:
            recommendations = topPopRecommender_removeSeen.recommend(user_id, at=10)
        else:
            recommendations = recommender.recommend(user_id, at=10)
        recommendations = utils.trim(recommendations)
        f.write(str(user_id) + "," + recommendations + "\n")



