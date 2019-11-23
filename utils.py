import scipy.sparse as sps
import random as rand


# Split input data into tuples, assuming 3 columns
def rowSplit(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = 1

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
        print(line)
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
    file = open(path, 'r')
    file.seek(9)
    print("Opened: " + path)

    column = []

    for line in file:
        column.append(int(line))

    return column

def get_features(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return featuresList

def get_entity(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return entityList


def create_test_matrix(path,offset):
    interactionList, userList, itemList  = create_tuples(path, offset)
    URM_csr = sps.coo_matrix(interactionList, (userList, itemList)).tocsr()

    # Build a test set
    test_mask = []
    for user in set(userList):
        lowBound = URM_csr.indptr[user]
        highBound = URM_csr.indptr[user + 1] - 1
        toRemove = rand.randint(lowBound, highBound)
        URM_csr.data[toRemove] = 0
        test_mask.append((user, itemList[toRemove], 1))
    URM_csr.eliminate_zeros()

    interactionList_test, userList_test, itemList_test  = zip(*test_mask)
    userList_test = list(userList_test)
    itemList_test = list(itemList_test)
    interactionList_test = list(interactionList_test)

    return sps.coo_matrix((interactionList_test, (userList_test, itemList_test))).tocsr()