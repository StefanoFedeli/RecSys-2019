# Split input data into tuple
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def rowSplit(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])  # user
    split[1] = int(split[1])  # feature
    split[2] = truncate(float(split[2]), 5)  # price

    return tuple(split)


def find_key(dictionary, target):
    for key, value in dictionary.items():
        if value == target:
            return key


prices_table = open("dataset/data_ICM_price.csv", 'r')
prices_table.seek(14)

prices_tuples = []

print("Fetching data from memory...")
numberInteractions = 0
for line in prices_table:
    numberInteractions += 1
    prices_tuples.append(rowSplit(line))

print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

itemList, featuresList, pricesList = zip(*prices_tuples)
itemList = list(itemList)
featuresList = list(featuresList)
pricesList = list(pricesList)

print(len(pricesList))
nonDuplicates = set(pricesList)
print(len(nonDuplicates))
nonDuplicates = list(nonDuplicates)

'''
distribution = np.zeros(len(nonDuplicates))
for i in range(0, len(nonDuplicates)):
    for element in assetsList:
        if nonDuplicates[i]==element:
            distribution[i] = distribution[i] + 1


print(distribution)
'''

myDictionary = {}
for i in range(0, len(nonDuplicates)):
    myDictionary[i]=nonDuplicates[i]
print(myDictionary)

with open("refinedDataSet/ICM_prices.csv", 'w') as f:
    f.write("row,col,data\n")
    for i in range(0, len(pricesList)):
        first = str(itemList[i])
        second = str(find_key(myDictionary, pricesList[i]))
        f.write(first+','+second+",1.0\n")