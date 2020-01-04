import numpy as np
import scipy.sparse as sps
import utils_new as util
import External_Libraries.Recommender_utils as mauri
import External_Libraries.Notebooks_utils.evaluation_function as eval


def normalize(URM_t, minI, maxI):
    MaxD = max(URM_t.data)
    MinD = min(URM_t.data)
    URM = URM_t.copy()
    print(minI,maxI)
    for i in range(URM.nnz):
        URM.data[i] = URM.data[i]*(maxI-minI)/(MaxD-MinD)
    return URM


class Reccomender(object):

    def __init__(self, URM, URM_t):
        self.URM = URM
        self.test = URM_t

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
        start_pos = self.test.indptr[user_id]
        end_pos = self.test.indptr[user_id + 1]

        user_profile = self.test.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


URM_1 = sps.csr_matrix(sps.load_npz("../../../Dataset/similarities/CB-Sim.npz"))
URM_2 = sps.csr_matrix(sps.load_npz("../../../Dataset/similarities/Col-Sim.npz"))
URM_3 = sps.csr_matrix(sps.load_npz("../../../Dataset/similarities/Slim-Sim.npz"))
URM_4 = normalize(URM_3,min(min(URM_1.data), min(URM_2.data)),max(max(URM_1.data), max(URM_2.data)))
URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
print(max(URM_1.data))
print(max(URM_2.data))
print(max(URM_3.data))
print(min(URM_1.data))
print(min(URM_2.data))
print(min(URM_3.data))
reccomender = Reccomender(URM_all,URM_all)
targetUsers = util.get_target_users("../../../dataset/target_users_cold.csv")
print("OK1")
#for i in range(0,3):
#    for j in range (0,3):
#        for z in range(0,3):
#            for norm in (False,True):
cumulative_precision = 0.0
cumulative_recall = 0.0
cumulative_MAP = 0.0
num_eval = 0
a = 0.4
b = 0.7
c = 0.9
norm = False
if norm:
    similarity_matrix = mauri.similarityMatrixTopK(a*URM_1 + b*URM_2 + c*URM_4, k=25)
else:
    similarity_matrix = mauri.similarityMatrixTopK(a * URM_1 + b * URM_2 + c * URM_3)
print("OK2")
reccomender.fit(similarity_matrix)
print("OK3")
for user in range(0):
    if num_eval % 1000 == 0:
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

    num_eval += 1
    cumulative_precision += eval.precision(is_relevant, relevant_items)
    cumulative_recall += eval.recall(is_relevant, relevant_items)
    cumulative_MAP += eval.MAP(is_relevant, relevant_items)

print("OK4")
with open("../../../Outputs/SSlim-cold.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in targetUsers:
        if user_id % 750 == 0:
            print("Evaluated user {} of {}".format(user_id, len(targetUsers)))
        f.write(str(user_id) + "," + util.trim(np.array(reccomender.recommend(user_id))) + "\n")
util.compare_csv("../../../Outputs/truth2.csv", "../../../Outputs/SSlim.csv")

print(cumulative_MAP)
cumulative_precision /= num_eval
cumulative_recall /= num_eval
cumulative_MAP /= num_eval


print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP@10 = {:.4f}".format(
    cumulative_precision, cumulative_recall, cumulative_MAP))

''''
with open("../../../Outputs/SSlim.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in goodUsers:
        f.write(str(user_id) + "\n")
 '''
result_dict = {
    "precision": cumulative_precision,
    "recall": cumulative_recall,
    "MAP": cumulative_MAP,
    "ALPHA": a,
    "BETA": b,
    "GAMMA": c,
    "NORM": norm
}
print(result_dict)

                #with open("../../../Outputs/combTest.txt", 'a+') as f:
                #    f.write(str(result_dict) + "\n")



