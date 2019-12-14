import scipy.sparse as sps
import numpy as np
import utils_new as utils
import pickle
import random

user_list = list(range(0,30911))
users_dict = dict()
print(len(user_list))
print(min(user_list))
print(max(user_list))


def select1(list_rec):
    values_all = []
    for rec in list_rec:
        for elem in rec:
            values_all.append(elem)
    values_set = set(values_all)
    rankings = {}
    for item in values_set:
        for i in range(0, len(values_all)):
            try:
                if values_all[i] == item:
                    rankings[item] += (10 - i%10)
            except KeyError:
                if values_all[i] == item:
                    rankings[item] = (10 - i%10)
    ascending_rank = list(sorted(rankings.items(), key=lambda x: x[1]))
    descending_rank = []
    for num in range(0, 10):
        descending_rank.append(ascending_rank[len(values_set)-num-1][0])
    print(descending_rank)
    return (descending_rank)

def select2(list_rec):
    start = random.randint(1, len(list_rec)) - 1
    #start = 0
    toReturn = []
    riga = (start + 1) % len(list_rec)
    col = 0
    for i in range(0, 10):

        while list_rec[riga][col] in toReturn:
            #print(riga, col, start)
            if riga % len(list_rec) == start:
                col += 1
            riga = (riga + 1) % len(list_rec)
        toReturn.append(list_rec[riga][col])
        if riga % len(list_rec) == start:
            col += 1
        riga = (riga + 1) % len(list_rec)
    print(toReturn)
    return toReturn

class TopPopRecommender(object):

    def fit(self, URM_train):
        self.URM_train = URM_train

        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            unseen_items = self.popularItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popularItems[0:at]
        return recommended_items

def create_recommendations():
    interaction_age, user_age, age = utils.create_tuples("../../../Dataset/UCM_age.csv",14)
    interaction_region, user_region, region = utils.create_tuples("../../../Dataset/UCM_region.csv",14)
    counter = 0
    interactions = interaction_age + interaction_region
    users = user_age + user_region
    features = age + region
    for u in user_list:
        try:
            index = user_age.index(u)
            target_age = age[index]
        except ValueError:
            target_age = -1
        try:
            index = user_region.index(u)
            target_region = region[index]
        except ValueError:
            target_region = -1
        if target_age>=0 and target_region>=0:
            key = int(str(target_age)+str(target_region))
            if key not in users_dict:
                users_dict[key] = list()
            users_dict[key].append(u)
            user_list.remove(u)
        elif target_age<0 and target_region>=0:
            for i in range(0, max(age)):
                key = int(str(i)+str(target_region))
                if key not in users_dict:
                    users_dict[key] = list()
                users_dict[key].append(u)
            user_list.remove(u)
        elif target_age>=0 and target_region<0:
            for i in range(min(region), max(region)):
                key = int(str(target_age)+str(i))
                if key not in users_dict:
                    users_dict[key] = list()
                users_dict[key].append(u)
            user_list.remove(u)

    users_dict[0] = user_list
    print(users_dict.keys())
    print(len(users_dict.keys()))

    recommendations = {}

    i = 0
    for key in users_dict.keys():
        if i % 7 == 0:
            print('{0}/70 Done'.format(i))
        shape = (30911, 18495)
        URM_cluster = utils.create_coo("../../../Dataset/URM.csv", users_dict.get(key), shape = shape)
        URM_cluster = URM_cluster.tocsr()
        recommender = TopPopRecommender()
        recommender.fit(URM_cluster)
        recommendations[key] = recommender
        print('categoria {0} ha {1} utenti'.format(key,len(users_dict.get(key))))
        i += 1

    users_freeze = utils.get_target_users("../../../Dataset/target_users.csv")
    outputs = {}

    for u in users_freeze:
        outputs[u] = []
        try:
            index = user_age.index(u)
            target_age = age[index]
        except ValueError:
            target_age = -1
        try:
            index = user_region.index(u)
            target_region = region[index]
        except ValueError:
            target_region = -1

        if target_age>=0 and target_region>=0:
            key = int(str(target_age)+str(target_region))
            outputs[u].append(recommendations.get(key).recommend(u))
        elif target_age<0 and target_region>=0:
            for i in range(0, max(age)):
                key = int(str(i)+str(target_region))
                outputs[u].append(recommendations.get(key).recommend(u))
        elif target_age>=0 and target_region<0:
            for i in range(min(region), max(region)):
                key = int(str(target_age)+str(i))
                outputs[u].append(recommendations.get(key).recommend(u))

    with open('../../../Dataset/topPopular.p', 'wb') as fp:
        pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)





#create_recommendations()
URM_all = sps.load_npz("../../../Dataset/data_all.npz")
URM_all = sps.csr_matrix(URM_all)

with open('../../../Dataset/topPopular.p', 'rb') as fp:
    outputs = pickle.load(fp)
f = open("../../../Outputs/TopPop_freeze.csv", 'w')
for u in utils.get_target_users("../../../Dataset/target_users_freeze.csv"):
    if u in outputs.keys():
        if len(outputs[u])==1:
            f.write(str(u) + "," + utils.trim(outputs[u][0]) + "\n")
        else:
            print('user want MIXER')
            f.write(str(u) + "," + utils.trim(select1(outputs[u])) + "\n")

