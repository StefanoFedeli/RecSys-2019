import random
import scipy.sparse as sps
import numpy as np
import utils_new as util
import External_Libraries.Notebooks_utils.evaluation_function as eval

def generate_rec(prediction, user_id):
    recommendation = []
    for i in range(10):
        added = False
        while not added:
            choice = random.randint(0,RecSys["truth"])
            for id,value in enumerate(RecSys.values(),0):
                if choice < value:
                    choice = prediction[id][user_id][0]
                    recommendation.append(choice)
                    added = True
                    for i in range(len(RecSys.keys())):
                        try:
                            prediction[i][user_id].remove(choice)
                        except ValueError:
                            len(prediction[i][user_id])
                break
    return recommendation[:10]

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
    recommendations = {}

    for user in target_users:
        if num_eval % 5000 == 0:
            print("Evaluated user {} of {}".format(num_eval, len(target_users)))

        start_pos = URM_test.indptr[user]
        end_pos = URM_test.indptr[user + 1]
        relevant_items = np.array([0])
        if end_pos - start_pos > 0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            # print(relevant_items)
            recommendations[user] = generate_rec(userList,user)
            is_relevant = np.in1d(recommendations[user], relevant_items, assume_unique=True)
        else:
            # num_eval += 1
            is_relevant = np.array([False, False, False, False, False, False, False, False, False, False])

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
        "RecSys": recommendations
    }
    return result_dict


RecSys = {
    "CBI": 0.005349698161970339,
    "Coll_I": 0.045073906922023704,
    "Coll_U": 0.24895662404289282,
    "slim-e2-75-80": 0.07087178151941191,
    "LightFM": 0.0048948099288318044,
    #"SSLIM": 0.0,
    "truth": 0.03210000000000000,
    #"TopPop": 0.00848452236301567,
}
userList = []

for idx, file in enumerate(RecSys.keys(),0):
    print(idx,file)
    userList.append({})
    seek = 19
    if "slim" in file:
        seek -= 1
    file = open("../../../Outputs/"+file+".csv")
    file.seek(seek)
    for line in file:
        split = line.split(",")
        userList[idx][(int(split[0]))] = list(map(int, split[1].split()))

sum = 0
for u in RecSys.values():
    sum += u
for key, value in RecSys.items():
    print(key,value)
    RecSys[key] = int(value*100/sum)
print(" ")
sum = 0
for key, value in RecSys.items():
    print(key,value)
    RecSys[key] = value + sum
    sum = RecSys[key]
print(RecSys)

URM_test = sps.csr_matrix(sps.load_npz("../../../dataset/data_test.npz"))
URM_val = sps.csr_matrix(sps.load_npz("../../../dataset/data_validation.npz"))
targetUsers = util.get_target_users("../../../dataset/target_users_other.csv")

print("TESTING")
res_test = run(targetUsers, URM_test)
print(res_test["MAP"])
#print(res_test)
#print("VALIDATION")
#res_val = run(targetUsers, URM_val)
#print(res_val["MAP"])
#print(res_val)

#print((res_val["MAP"]+res_test["MAP"])/2)