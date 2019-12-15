import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import utils_new as utils
import scipy.sparse as sps
import External_Libraries.Notebooks_utils.evaluation_function as eval
import random

utils.compare_csv("../../../Outputs/TopPop_freeze.csv", "../../../Outputs/LightFM_topPop_1_9600_all.csv")
exit()

URM_all = sps.coo_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.coo_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))

features1 = utils.get_second_column("../../../Dataset/UCM_age.csv")
features2 = utils.get_second_column("../../../Dataset/UCM_region.csv")
features = features1 + features2

entity1 = utils.get_first_column("../../../Dataset/UCM_age.csv")
entity2 = utils.get_first_column("../../../Dataset/UCM_region.csv")
entities = entity1 + entity2
ones = np.ones(len(features))
UCM_all = sps.coo_matrix((ones, (entities, features)))
UCM_all = UCM_all.tocsr()

features1 = utils.get_second_column("../../../Dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../Dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../Dataset/ICM_asset.csv")
features = features1 + features2 + features3
items1 = utils.get_first_column("../../../Dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../Dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../Dataset/ICM_asset.csv")
items = items1 + items2 + items3
ones = np.ones(len(features))
n_items = URM_train.shape[1]
n_tags = max(features) + 1
ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()

cold_u = URM_all.shape[0]-len(utils.get_target_users('../../../Dataset/target_users_freeze.csv'))
target_users = utils.get_target_users("../../../Dataset/target_users_freeze.csv")

print('The dataset has %s users (%s warm) and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (URM_train.shape[0], cold_u, URM_train.shape[1], URM_test.getnnz(), URM_train.getnnz()))
URM_all = sps.csr_matrix(URM_all)
URM_train = sps.csr_matrix(URM_train)
class Recommender(object):

    def __init__(self, URM, model, URM_train):
        self.URM = URM
        self.model = model
        self.test=URM_train

    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = model.predict(user_id, np.array(list(range(n_items))), user_features=UCM_all, item_features=ICM_all)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.test.indptr[user_id]
        end_pos = self.test.indptr[user_id + 1]

        user_profile = self.test.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


COMPONENTS = 1
NUM_EPOCHS = 9600
ITEM_ALPHA = 1e-6
LEARNING = 'adadelta'
LEARNING_RATE = 1e-5
LOSS = 'warp'

# Let's fit a WARP model: these generally have the best performance.
model = LightFM(loss=LOSS,
                item_alpha=ITEM_ALPHA,
                no_components=COMPONENTS,
                learning_schedule=LEARNING,
                learning_rate=LEARNING_RATE)

print("Currently using LOSS:{0}, COMPONENTS:{1}, LEARNING:{2}, RATE:{3}".format(LOSS, COMPONENTS, LEARNING, LEARNING_RATE))

# Run 3 epochs and time it.
model = model.fit(URM_all, user_features=UCM_all, item_features=ICM_all, epochs=NUM_EPOCHS, verbose=True)
recommender = Recommender(URM_all, model, URM_train)

cumulative_precision = 0.0
cumulative_recall = 0.0
cumulative_MAP = 0.0
num_eval = 0

for user in target_users:
    if num_eval % 7500 == 0:
        print("Evaluated user {} of {}".format(num_eval, len(target_users)))

    start_pos = URM_test.indptr[user]
    end_pos = URM_test.indptr[user + 1]
    relevant_items = np.array([0])
    if end_pos - start_pos > 0:

        relevant_items = URM_test.indices[start_pos:end_pos]
        # print(relevant_items)

        is_relevant = np.in1d(recommender.recommend(user), relevant_items, assume_unique=True)
    else:
        # num_eval += 1
        is_relevant = np.array([False, False, False, False, False, False, False, False, False, False])

    #if True in is_relevant[:3]:
        #goodUsers.append(user)
    num_eval += 1
    cumulative_precision += eval.precision(is_relevant, relevant_items)
    cumulative_recall += eval.recall(is_relevant, relevant_items)
    cumulative_MAP += eval.MAP(is_relevant, relevant_items)

cumulative_precision /= num_eval
cumulative_recall /= num_eval
cumulative_MAP /= num_eval

result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP
}
print(result_dict)


with open("../../../Outputs/LightFM_topPop_1_9600_all.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in target_users:
        f.write(str(user_id) + "," + utils.trim(np.array(recommender.recommend(user_id))) + "\n")
