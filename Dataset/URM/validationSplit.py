import scipy.sparse as sps
import utils_new as util
import numpy as np

# Uncomment to generate new dataset
#util.createDataset(".")

URM = sps.load_npz("data_all.npz")
URM = URM.tocsr()
cold_users = []
targetUsers = util.get_target_users("../target_users.csv", seek=8)
print(len(targetUsers))
print(max(targetUsers))
cold_user_mask = np.ediff1d(URM.indptr) == 0
user_mask = np.ediff1d(URM.indptr) < 3
#print(len(user_mask[user_mask == True]))
#exit()
for i in range(len(user_mask)):
    if user_mask[i] and i in targetUsers and not cold_user_mask[i]:
        cold_users.append(i)
        targetUsers.remove(i)

with open("../target_users_lukewarm.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in cold_users:
        f.write(str(user_id) + "\n")

'''''
with open("target_users_other.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in targetUsers:
        f.write(str(user_id) + "\n")
'''
print(len(cold_users))
print(len(targetUsers))
print(len(cold_users) + len(targetUsers))