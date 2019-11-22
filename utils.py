import scipy.sparse as sps


# Split input data into tuples, assuming 3 columns
def rowSplit(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = 1

    return tuple(split)

# Creates a coo given the path of a 3 columns csv
def create_coo(path):
    file = open(path, 'r')
    file.seek(14)
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

    return sps.coo_matrix((interactionList, (entityList, featuresList)))

def get_first_column(path):
    file = open(path, 'r')
    file.seek(9)
    print("Opened: " + path)

    column = []

    for line in file:
        column.append(int(line))

    return column