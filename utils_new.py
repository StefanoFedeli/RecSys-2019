import scipy.sparse as sps
import External_Libraries.Zeus.split_train_validation_leave_k_out as split_data

# Split input data into tuples, assuming 3 columns
def rowSplit(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    return tuple(split)

# Creates a coo given the path of a 3 columns dataset
def create_tuples(path, offset, filter = None):
    file = open(path, 'r')
    file.seek(offset)
    print("Opened: " + path)


    tuples = []

    print("Fetching data from memory...")
    numberInteractions = 0
    for line in file:
        numberInteractions += 1
        split = rowSplit(line)
        if filter:
            if split[0] in filter:
                tuples.append(split)
        else:
            tuples.append(split)

    print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

    entityList, featuresList, interactionList = zip(*tuples)
    entityList = list(entityList)
    featuresList = list(featuresList)
    interactionList = list(interactionList)

    return interactionList, entityList, featuresList

def create_coo(path, filter = None, shape = None):
    interactionList, entityList, featuresList = create_tuples(path, 14, filter)
    if shape:
        return sps.coo_matrix((interactionList, (entityList, featuresList)), shape)
    else:
        return sps.coo_matrix((interactionList, (entityList, featuresList)))

def get_first_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return entityList

def get_second_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return featuresList

def get_third_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return interactionList

def get_target_users(path):
    file = open(path, 'r')
    file.seek(9)
    print("Opened: " + path)

    column = []

    for line in file:
        column.append(int(line))

    return column

def trim(array):
    string = str(array)
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace(",", "")
    split = str(string).split(" ")
    split = list(filter(None, split))
    return split[0] + " " + split[1] + " " + split[2] + " " + split[3] + " " + split[4] + " " + split[5] + " " + split[6] + " " + split[7] + " " + split[8] + " " + split[9]

def createDataset(relPath):
    URM_raw = create_coo(relPath + "/URM.csv")
    sps.save_npz(relPath + "/data_all.npz", URM_raw)
    URM_raw, URM_test = split_data.split_train_leave_k_out_user_wise(URM_raw, use_validation_set=False)
    URM_train, URM_validation = split_data.split_train_leave_k_out_user_wise(URM_raw, use_validation_set=False)
    sps.save_npz(relPath + "/data_train.npz", URM_train)
    sps.save_npz(relPath + "/data_test.npz", URM_test)
    sps.save_npz(relPath + "/data_validation.npz", URM_validation)

def compare_csv(csv1, csv2):
    f1 = open(csv1, 'r')
    f2 = open(csv2, 'r')
    f1.seek(19)
    f2.seek(19)
    userList1 = {}
    userList2 = {}
    for line in f1:
        split = line.split(",")
        userList1[(int(split[0]))] = list(map(int, split[1].split()))
    for line in f2:
        split = line.split(",")
        userList2[(int(split[0]))] = list(map(int, split[1].split()))
    cumulativeError = 0
    for user in userList2.keys():
        list1 = userList1.get(user)
        list2 = userList2.get(user)
        localError = 0
        for item in list2:
            try:
                localError = localError + list1.index(item)-list2.index(item)
            except ValueError:
                localError = localError + 10
        cumulativeError = cumulativeError + localError
    averageCumulativeError = cumulativeError/len(userList1.keys())
    similarity = 100 - averageCumulativeError
    print("Average similarity " + str(similarity) + "%")

#compare_csv("Outputs/truth.csv", "Outputs/Sslim.csv")