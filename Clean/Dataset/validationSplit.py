import scipy.sparse as sps
import Clean.utils_new as util

# Uncomment to generate new dataset
# util.createDataset(".")

URM = sps.load_npz("data_train.npz")
URM = URM.tocsr()
cold_users = []
targetUsers = util.get_target_users("target_users.csv")
print(len(targetUsers))
for user in targetUsers:
    if URM.indptr[user]-URM.indptr[user+1] == 0:
        cold_users.append(user)
        targetUsers.remove(user)

with open("target_users_cold.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in cold_users:
        f.write(str(user_id) + "\n")

with open("target_users_other.csv", 'w') as f:
    f.write("user_id\n")
    for user_id in targetUsers:
        f.write(str(user_id) + "\n")

print(len(cold_users))
print(len(targetUsers))
print(len(cold_users) + len(targetUsers))