import numpy as np
import scipy.sparse as sps
import utils_new as util
import External_Libraries.Notebooks_utils.evaluation_function as eval

def mergeCSV(filelist, seek=19):
    userList = {}
    # Be aware that if a user is in both files then is overwritten
    for file in filelist:
        file.seek(seek)
        for line in file:
            split = line.split(",")
            userList[(int(split[0]))] = list(map(int, split[1].split()))
    return userList

def run(target_users, URM_test):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    recommendations = mergeCSV([open("../Outputs/TopPop_cold.csv", "r"),
                                open("../Outputs/TopPop_freeze.csv", "r")
                                ], seek=18)
    goodUsers = []

    for user in target_users:
        if num_eval % 5000 == 0:
            print("Evaluated user {} of {}".format(num_eval, len(target_users)))

        start_pos = URM_test.indptr[user]
        end_pos = URM_test.indptr[user + 1]
        relevant_items = np.array([0])
        if end_pos - start_pos > 0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            # print(relevant_items)

            is_relevant = np.in1d(recommendations[user], relevant_items, assume_unique=True)
        else:
            num_eval += 1
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
        "Users": goodUsers,
        "RecSys": recommendations
    }
    return result_dict


URM_test = sps.csr_matrix(sps.load_npz("../Dataset/URM/data_test.npz"))
#URM_val = sps.csr_matrix(sps.load_npz("../Dataset/old/data_validation.npz"))
targetUsers = util.get_target_users("../Dataset/target_users_cold.csv", seek=8)

print("TESTING")
res_test = run(targetUsers, URM_test)
print(res_test["MAP"])
"""
print(res_test)
print("VALIDATION")
res_val = run(targetUsers, URM_val)
print(res_val["MAP"])
#print(res_val)

print((res_val["MAP"]+res_test["MAP"])/2)
"""
"""''
with open("../../../Outputs/HybridSte.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in targetUsers:
        #print(user_id)
        f.write(str(user_id) + "," + util.trim(np.array(res_test["RecSys"][user_id])) + "\n")

"""
#util.compare_csv("../../../Outputs/truth2.csv", "../../../Outputs/HappyNewHybrid_0.8_0.25_0.15_0055.csv")
