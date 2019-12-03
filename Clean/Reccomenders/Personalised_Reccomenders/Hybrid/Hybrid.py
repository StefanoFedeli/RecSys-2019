import numpy as np
import scipy.sparse as sps
from Clean.External_Libraries.Zeus.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise as train_test_leaveoneout


def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score



def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score



def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    # print(is_relevant)
    # print(relevant_items)
    # print(p_at_k)

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score





cumulative_precision = 0.0
cumulative_recall = 0.0
cumulative_MAP = 0.0
num_eval = 0

URM_test = sps.load_npz("data_test.npz")

URM_test = sps.csr_matrix(URM_test)

file = open("../../dataset/slim.csv", "r")
#file = open("../../dataset/matrixFact.csv", "r")
#file = open("../Content Based/output.csv", "r")
# n_users = 1

for line in file: #range(4):
    if num_eval % 5000 == 0:
        print("Evaluated user {} of {}".format(num_eval, URM_test.shape[1]))

    split = line.split(",")
    user_id = int(split[0])
    recommended_items = list(map(int, split[1].split()))

    start_pos = URM_test.indptr[user_id]
    end_pos = URM_test.indptr[user_id+1]

    if end_pos-start_pos > 0:

        relevant_items = URM_test.indices[start_pos:end_pos]
        # print(relevant_items)
        num_eval += 1

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        cumulative_precision += precision(is_relevant, relevant_items)
        cumulative_recall += recall(is_relevant, relevant_items)
        cumulative_MAP += MAP(is_relevant, relevant_items)

cumulative_precision /= num_eval
cumulative_recall /= num_eval
cumulative_MAP /= num_eval

print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP@10 = {:.4f}".format(
    cumulative_precision, cumulative_recall, cumulative_MAP))

result_dict = {
    "precision": cumulative_precision,
    "recall": cumulative_recall,
    "MAP": cumulative_MAP,
}
print(result_dict)