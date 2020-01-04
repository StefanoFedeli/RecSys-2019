import numpy as np
import External_Libraries.Notebooks_utils.evaluation_function as eval

def evaluate(target_users, recommender, URM_test):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    goodUsers = []

    for user in target_users:
        '''
        if num_eval % 5000 == 0:
            print("Evaluated user {} of {}".format(num_eval, len(target_users)))
        '''

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

        if True in is_relevant[:3]:
            goodUsers.append(user)
        num_eval += 1
        cumulative_precision += eval.precision(is_relevant, relevant_items)
        cumulative_recall += eval.recall(is_relevant, relevant_items)
        cumulative_MAP += eval.MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP@10 = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
        "Users": goodUsers
    }
    return result_dict