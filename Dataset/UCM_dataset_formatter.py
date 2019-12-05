import utils_new as utils

UCM_age_features = utils.get_second_column("Original_dataset/data_UCM_age.csv")
UCM_region_features = utils.get_second_column("Original_dataset/data_UCM_region.csv")

UCM_age_features = set(UCM_age_features)
UCM_region_features = set(UCM_region_features)

UCM_age_features = list(UCM_age_features)
UCM_region_features = list(UCM_region_features)

age_length = len(UCM_age_features)
region_length = len(UCM_region_features)

length1 = age_length
length2 = length1 + region_length

features_dictionary = {}
for i in range(0, length1):
    features_dictionary[i] = UCM_age_features[i]
for i in range(length1, length2):
    features_dictionary[i] = UCM_region_features[i-length1]
key_list = list(features_dictionary.keys())
val_list = list(features_dictionary.values())

print(features_dictionary)

with open("../Original_dataset/data_UCM_age.csv") as f1:
    with open("UCM_age.csv", 'w') as f2:
        f2.write("row,col,data\n")
        f1.seek(14)
        for line in f1:
            split = line.split(",")
            print(split)
            split[2] = split[2].replace("\n", "")
            entity = int(split[0])
            feature = key_list[val_list.index(int(split[1]))]
            interaction = 1.0
            f2.write(str(entity) + ", " + str(feature) + ", " + str(interaction) + "\n")

with open("../Original_dataset/data_UCM_region.csv") as f1:
    with open("UCM_region.csv", 'w') as f2:
        f2.write("row,col,data\n")
        f1.seek(14)
        for line in f1:
            split = line.split(",")
            print(split)
            split[2] = split[2].replace("\n", "")
            entity = int(split[0])
            feature = key_list[val_list.index(int(split[1]))]
            interaction = 1.0
            f2.write(str(entity) + ", " + str(feature) + ", " + str(interaction) + "\n")
