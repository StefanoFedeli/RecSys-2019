import scipy.sparse as sps
import numpy as np
import utils_new as utils
import pickle
import random
from sklearn.cluster import KMeans


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
    #print(descending_rank)
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
    interaction_age, user_age, age = utils.create_tuples("../../Dataset/UCM/UCM_age.csv", 13)
    interaction_region, user_region, region = utils.create_tuples("../../Dataset/UCM/UCM_region.csv", 13)
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

    kmeans = KMeans(n_clusters=2)
    resKNN = kmeans.fit_predict(sps.coo_matrix(utils.create_coo("../../Dataset/URM/URM.csv", shape=(30911, 18495))))
    for u in user_list:
        key = 0-resKNN[u]
        if key not in users_dict:
            users_dict[key] = list()
        users_dict[key].append(u)
    '''
    for k in users_dict.keys():
        print(k)
        print(len(users_dict.get(k)))

    exit()
    '''
    recommendations = {}

    i = 0
    for key in users_dict.keys():
        if i % 7 == 0:
            print('{0}/{1} Done'.format(i, len(users_dict.keys())))
        shape = (30911, 18495)
        URM_cluster = utils.create_coo("../../Dataset/URM/URM.csv", users_dict.get(key), shape = shape)
        URM_cluster = URM_cluster.tocsr()
        recommender = TopPopRecommender()
        recommender.fit(URM_cluster)
        recommendations[key] = recommender
        print('categoria {0} ha {1} utenti'.format(key,len(users_dict.get(key))))
        i += 1

    users_cold = utils.get_target_users("../../Dataset/target_users_cold.csv", seek=8)
    outputs = {}

    for u in users_cold:
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
            for i in range(0, max(age)+1):
                key = int(str(i)+str(target_region))
                outputs[u].append(recommendations.get(key).recommend(u))
        elif target_age>=0 and target_region<0:
            for i in range(min(region), max(region)+1):
                key = int(str(target_age)+str(i))
                outputs[u].append(recommendations.get(key).recommend(u))
        elif target_age < 0 and target_region < 0:
            if u in users_dict[0]:
                outputs[u].append(recommendations.get(0).recommend(u))
            else:
                outputs[u].append(recommendations.get(-1).recommend(u))
        #print(u)
    print("DONE!")

    with open('../../Dataset/topPopularRecommendations.p', 'wb') as fp:
        pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return outputs


#output = create_recommendations()
#exit()
with open('../../Dataset/topPopularRecommendations.p', 'rb') as fp:
    output = pickle.load(fp)
f = open("../../Outputs/TopPop_freeze.csv", 'w')
f.write("user_id,item_list\n")
for u in utils.get_target_users("../../Dataset/target_users_freeze.csv",seek=8):
    if u in output.keys():
        if len(output[u])==1:
            f.write(str(u) + "," + utils.trim(output[u][0]) + "\n")
        else:
            #print('user want MIXER')
            f.write(str(u) + "," + utils.trim(select1(output[u])) + "\n")
    else:
        print("WE HAVE A PROBLEM {0} IS NOT HERE".format(u))

