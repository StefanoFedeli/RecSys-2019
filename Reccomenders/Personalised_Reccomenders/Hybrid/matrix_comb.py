import numpy as np
import scipy.sparse as sps
import utils_new as util


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
    # Be aware that if a user is in both files then is overwritten
    for line in file1:
        split = line.split(",")
        userList[(int(split[0]))] = list(map(int, split[1].split()))
    for line in file2:
        split = line.split(",")
        userList[(int(split[0]))] = list(map(int, split[1].split()))
    return userList

class Reccomender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, similarity_matrix):
        self.W_sparse = similarity_matrix

    def recommend(self, user_id, at=10, exclude_seen=True):
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

cumulative_precision = 0.0
cumulative_recall = 0.0
cumulative_MAP = 0.0
num_eval = 0
URM_1 = sps.load_npz("../../../Dataset/Similarity.npz")
URM_2 = sps.load_npz("../../../Dataset/slim-similarity.npz")
URM_1 = sps.csr_matrix(URM_1)
URM_2 = sps.csr_matrix(URM_2)
URM_test = sps.load_npz("../../../Dataset/data_test.npz")
URM_test = sps.csr_matrix(URM_test)
print(URM_1.shape)
print(URM_2.shape)
print(max(URM_1.data))
print(max(URM_2.data))
print(min(URM_1.data))
print(min(URM_2.data))
reccomender = Reccomender(sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz")))
targetUsers = util.get_target_users("../../../dataset/target_users_other.csv")
goodUsers = []
for i in range(0,10):
    num_eval = 0
    alfa = 1 - i*0.1
    similarity_matrix = alfa*URM_1 + (1-alfa)*URM_2
    reccomender.fit(similarity_matrix)
    for user in targetUsers:
        if num_eval % 200000 == 0:
            print("Evaluated user {} of {}".format(num_eval, len(targetUsers)))

        start_pos = URM_test.indptr[user]
        end_pos = URM_test.indptr[user + 1]
        relevant_items = np.array([0])
        if end_pos - start_pos > 0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            # print(relevant_items)

            is_relevant = np.in1d(reccomender.recommend(user), relevant_items, assume_unique=True)
        else:
            # num_eval += 1
            is_relevant = np.array([False, False, False, False, False, False, False, False, False, False])

        if True in is_relevant[:3]:
            goodUsers.append(user)
        num_eval += 1
        cumulative_precision += precision(is_relevant, relevant_items)
        cumulative_recall += recall(is_relevant, relevant_items)
        cumulative_MAP += MAP(is_relevant, relevant_items)

    ''' 
    with open("../../../Outputs/test_Sslim.csv", 'w') as f:
        f.write("user_id,item_list\n")
        for user_id in targetUsers:
            f.write(str(user_id) + "," + util.trim(np.array(reccomender.recommend(user_id))) + "\n")
    util.compare_csv("../../../Outputs/truth.csv","../../../Outputs/test_Sslim.csv")
    '''
    with open("../../../Outputs/Sslim.csv", 'w') as f:
        f.write("user_id,item_list\n")
        for user_id in targetUsers:
            f.write(str(user_id) + "," + util.trim(np.array(reccomender.recommend(user_id))) + "\n")
    util.compare_csv("../../../Outputs/truth.csv", "../../../Outputs/Sslim.csv")

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    '''  
    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP@10 = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))
    
    with open("../../../Outputs/UserSlim.csv", 'w') as f:
        f.write("user_id\n")
        for user_id in goodUsers:
            f.write(str(user_id) + "\n")
        '''
    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
        "ALPHA": alfa,
    }
    print(result_dict)


exit()


# file = open("../Content Based/output.csv", "r")
# n_users = 1
recommendations = mergeCSV( open("../../../Outputs/pytorch-e1-500-80-all.csv", "r"), open("../../../Outputs/TopPop_cold.csv", "r"))
targetUsers = util.get_target_users("../../../dataset/target_users.csv")
# targetUsers = util.get_target_users("../../../dataset/target_users_cold.csv")
goodUsers = []


