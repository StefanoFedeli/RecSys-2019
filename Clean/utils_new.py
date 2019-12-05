import scipy.sparse as sps
import Clean.External_Libraries.Zeus.split_train_validation_leave_k_out as split_data

# Split input data into tuples, assuming 3 columns
def rowSplit(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    return tuple(split)

# Creates a coo given the path of a 3 columns dataset
def create_tuples(path, offset):
    file = open(path, 'r')
    file.seek(offset)
    print("Opened: " + path)


    tuples = []

    print("Fetching data from memory...")
    numberInteractions = 0
    for line in file:
        numberInteractions += 1
        tuples.append(rowSplit(line))

    print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

    entityList, featuresList, interactionList = zip(*tuples)
    entityList = list(entityList)
    featuresList = list(featuresList)
    interactionList = list(interactionList)

    return interactionList, entityList, featuresList

def create_coo(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return sps.coo_matrix((interactionList, (entityList, featuresList)))

def get_first_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return featuresList

def get_second_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return entityList

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
