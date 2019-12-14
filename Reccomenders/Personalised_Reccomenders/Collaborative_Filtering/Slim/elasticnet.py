import utils_new as utils
import numpy as np

userList = {}
file = open("../../../../Outputs/slim-e2-75-80.csv", "r")
file.seek(18)
# Be aware that if a user is in both files then is overwritten
for line in file:
    split = line.split(",")
    userList[(int(split[0]))] = list(map(int, split[1].split()))

users = utils.get_target_users("../../../../Dataset/users_clusters/Slim.csv")
with open("../../../../Outputs/Slim.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(np.array(userList[user_id])) + "\n")