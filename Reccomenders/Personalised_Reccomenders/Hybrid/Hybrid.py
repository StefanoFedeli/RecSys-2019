import numpy as np
import scipy.sparse as sps
import utils_new as util


util.compare_csv("../../../Outputs/TopPop_freeze.csv", "../../../Outputs/LightFM_topPop_3_600.csv")
exit()


def precision(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


def recall(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


def MAP(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


def mergeCSV(file1, file2):
    userList = {}
    file1.seek(19)
    file2.seek(19)
    # Be aware that if a user is in both files then is overwritten
    for line in file1:
        split = line.split(",")
        userList[(int(split[0]))] = list(map(int, split[1].split()))
    for line in file2:
        split = line.split(",")
        userList[(int(split[0]))] = list(map(int, split[1].split()))
    return userList

cumulative_precision = 0.0
cumulative_recall = 0.0
cumulative_MAP = 0.0
num_eval = 0
URM_test = sps.load_npz("../../../dataset/data_test.npz")
URM_test = sps.csr_matrix(URM_test)

# file = open("../Content Based/output.csv", "r")
# n_users = 1
recommendations = mergeCSV(open("../../../Outputs/CollStd+NewTopPop.csv", "r"), open("../../../Outputs/LightFM_topPop.csv", "r"))
targetUsers = util.get_target_users("../../../dataset/target_users.csv")
# targetUsers = util.get_target_users("../../../dataset/target_users_cold.csv")
goodUsers = []

for user in targetUsers:
    if num_eval % 5000 == 0:
        print("Evaluated user {} of {}".format(num_eval, len(targetUsers)))

    start_pos = URM_test.indptr[user]
    end_pos = URM_test.indptr[user+1]
    relevant_items = np.array([0])
    if end_pos-start_pos > 0:

        relevant_items = URM_test.indices[start_pos:end_pos]
        # print(relevant_items)

        is_relevant = np.in1d(recommendations[user], relevant_items, assume_unique=True)
    else:
        #num_eval += 1
        is_relevant = np.array([False, False, False, False, False, False, False, False, False, False])

    if True in is_relevant[:3]:
        goodUsers.append(user)
    num_eval += 1
    cumulative_precision += precision(is_relevant, relevant_items)
    cumulative_recall += recall(is_relevant, relevant_items)
    cumulative_MAP += MAP(is_relevant, relevant_items)


with open("../../../Outputs/CollStd+LightFMTopPop.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in targetUsers:
        f.write(str(user_id) + "," + util.trim(np.array(recommendations[user_id])) + "\n")



cumulative_precision /= num_eval
cumulative_recall /= num_eval
cumulative_MAP /= num_eval

print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP@10 = {:.4f}".format(
    cumulative_precision, cumulative_recall, cumulative_MAP))

''' 
with open("../../../Outputs/UserSlim.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in goodUsers:
        f.write(str(user_id) + "\n")
'''
result_dict = {
    "precision": cumulative_precision,
    "recall": cumulative_recall,
    "MAP": cumulative_MAP,
}
print(result_dict)

util.compare_csv("../../../Outputs/truth.csv", "../../../Outputs/CollStd+LightFMTopPop.csv")
