import Clean.utils_new as utils

ICM_asset_features = utils.get_third_column("../Original_dataset/data_ICM_asset.csv")
ICM_price_features = utils.get_third_column("../Original_dataset/data_ICM_price.csv")
ICM_sub_class_features = utils.get_second_column("../Original_dataset/data_ICM_sub_class.csv")

ICM_asset_features = set(ICM_asset_features)
ICM_price_features = set(ICM_price_features)
ICM_sub_class_features = set(ICM_sub_class_features)

ICM_asset_features = list(ICM_asset_features)
ICM_price_features = list(ICM_price_features)
ICM_sub_class_features = list(ICM_sub_class_features)

asset_length = len(ICM_asset_features)
price_length = len(ICM_price_features)
sub_class_length = len(ICM_sub_class_features)

length1 = asset_length
length2 = length1 + price_length
length3 = length2 + sub_class_length

features_dictionary = {}
for i in range(0, length1):
    features_dictionary[i] = ICM_asset_features[i]
for i in range(length1, length2):
    features_dictionary[i] = ICM_price_features[i-length1]
for i in range(length2, length3):
    features_dictionary[i] = ICM_sub_class_features[i-length2]
key_list = list(features_dictionary.keys())
val_list = list(features_dictionary.values())

with open("../Original_dataset/data_ICM_asset.csv", 'r') as f1:
    with open("ICM_asset.csv", 'w') as f2:
        f2.write("row,col,data\n")
        f1.seek(14)
        for line in f1:
            split = line.split(",")
            print(split)
            split[2] = split[2].replace("\n", "")
            entity = int(split[0])
            feature = key_list[val_list.index(float(split[2]))]
            interaction = 1.0
            f2.write(str(entity) + ", " + str(feature) + ", " + str(interaction) + "\n")

with open("../Original_dataset/data_ICM_price.csv", 'r') as f1:
    with open("ICM_price.csv", 'w') as f2:
        f2.write("row,col,data\n")
        f1.seek(14)
        for line in f1:
            split = line.split(",")
            print(split)
            split[2] = split[2].replace("\n", "")
            entity = int(split[0])
            feature = key_list[val_list.index(float(split[2]))]
            interaction = 1.0
            f2.write(str(entity) + ", " + str(feature) + ", " + str(interaction) + "\n")

with open("../Original_dataset/data_ICM_sub_class.csv", 'r') as f1:
    with open("ICM_sub_class.csv", 'w') as f2:
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
