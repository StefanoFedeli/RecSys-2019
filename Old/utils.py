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
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return featuresList

def get_second_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return entityList

def get_third_column(path):
    interactionList, entityList, featuresList = create_tuples(path, 14)
    return interactionList