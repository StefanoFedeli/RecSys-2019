import scipy.sparse as sps
import numpy as np
import utils_new as utils

class TopPopRecommender(object):

    def fit(self, URM_train):
        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=10):
        recommended_items = self.popularItems[0:at]
        return recommended_items


URM_train = sps.load_npz("../../../Dataset/data_train.npz")
topPopRecommender = TopPopRecommender()
topPopRecommender.fit(URM_train)

users = utils.get_target_users("../../../Dataset/target_users_cold.csv")
with open("../../../Outputs/TopPop_cold.csv", 'w') as f:
    # f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(topPopRecommender.recommend(user_id, at=10)) + "\n")

