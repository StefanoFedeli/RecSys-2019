import scipy.sparse as sps
import numpy as np
import utils_new as utils

user_list = list(range(30910))
users = dict()




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


interaction, user, age = utils.create_tuples("../../../Dataset/UCM_age.csv",14)
counter = 0
for u in user:
    a = age[counter]
    if a not in users:
        users[a] = list()
    users[a].append(u)
    user_list.remove(u)
    counter += 1
users[10] = user_list
print(users.keys())
