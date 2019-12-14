import numpy as np
import scipy.sparse as sps
import utils_new as util
import External_Libraries.Notebooks_utils.evaluation_function as eval

''''
#util.compare_csv("../../../Outputs/TopPop_freeze.csv", "../../../Outputs/LightFM_topPop_3_600.csv")
CBI = util.get_target_users("../../../dataset/users_clusters/CBI.csv")
Coll_I = util.get_target_users("../../../dataset/users_clusters/Coll_I.csv")
Coll_U = util.get_target_users("../../../dataset/users_clusters/Coll_U.csv")
Slim = util.get_target_users("../../../dataset/users_clusters/Slim.csv")
for u in CBI:
    if u in Coll_I:
        print(u)
        Coll_I.remove(u)
    if u in Coll_U:
        print(u)
        Coll_U.remove(u)
    if u in Slim:
        print(u)
        Slim.remove(u)
for u in Coll_I:
    if u in Coll_U:
        print(u)
        Coll_U.remove(u)
    if u in Slim:
        print(u)
        Slim.remove(u)
for u in Slim:
    if u in Coll_U:
        print(u)
        Coll_U.remove(u)
with open("../../../Dataset/users_clusters/CBI.csv", 'w') as f:
    f.write("user_id\n")
    for i in CBI:
        f.write(str(i) + "\n")
with open("../../../Dataset/users_clusters/Coll_I.csv", 'w') as f:
    f.write("user_id\n")
    for i in Coll_I:
        f.write(str(i) + "\n")
with open("../../../Dataset/users_clusters/Coll_U.csv", 'w') as f:
    f.write("user_id\n")
    for i in Coll_U:
        f.write(str(i) + "\n")
with open("../../../Dataset/users_clusters/Slim.csv", 'w') as f:
    f.write("user_id\n")
    for i in Slim:
        f.write(str(i) + "\n")
exit()
'''
def mergeCSV(filelist):
    userList = {}
    # Be aware that if a user is in both files then is overwritten
    for file in filelist:
        file.seek(19)
        for line in file:
            split = line.split(",")
            userList[(int(split[0]))] = list(map(int, split[1].split()))
    return userList

def run(target_users, URM_test):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    '''
    recommendations = mergeCSV([open("../../../Outputs/truth.csv", "r"),
                                open("../../../Outputs/TopPop_freeze.csv", "r"),
                                open("../../../Outputs/Slim.csv", "r"),
                                open("../../../Outputs/Coll_U.csv", "r"),
                                open("../../../Outputs/Coll_I.csv", "r"),
                                open("../../../Outputs/CBI.csv", "r")
                               ])
    '''
    recommendations = mergeCSV([open("../../../Outputs/Coll_U.csv", "r")])
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
        "Users": goodUsers,
        "RecSys": recommendations
    }
    return result_dict


URM_test = sps.csr_matrix(sps.load_npz("../../../dataset/data_test.npz"))
URM_val = sps.csr_matrix(sps.load_npz("../../../dataset/data_validation.npz"))
targetUsers = util.get_target_users("../../../dataset/target_users.csv")

print("TESTING")
res_test = run(targetUsers, URM_test)
print(res_test["MAP"])
#print(res_test)
print("VALIDATION")
res_val = run(targetUsers, URM_val)
print(res_val["MAP"])
#print(res_val)


with open("../../../Dataset/users_clusters/Coll_U.csv", 'w') as f:
    f.write("user_id\n")
    for i in range(0, 30911):
        if i in res_test["Users"] and i in res_val["Users"]:
            f.write(str(i) + "\n")


'''
with open("../../../Outputs/PureHybrid.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in targetUsers:
        f.write(str(user_id) + "," + util.trim(np.array(res_test["RecSys"][user_id])) + "\n")
        
util.compare_csv("../../../Outputs/truth.csv", "../../../Outputs/PureHybrid.csv")
'''