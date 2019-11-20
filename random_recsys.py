import scipy.sparse as sps
import numpy as np
# import matplotlib.pyplot as pyplot
# import csv

# MY RECOMMENDED SYSTEM
class RandomRecommender(object):

    def fit(self, ICM_train):
        self.numSongs = ICM_train.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.numSongs, at)
        return recommended_items

# METRICS: Precision
def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


# EVALUATION
def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    # cumulative_recall = 0.0
    # cumulative_MAP = 0.0
    num_eval = 0

    for playlist_id in playList_unique:

        if num_eval % 10000 == 0:
            print("Progress {:.2f}%".format(num_eval/len(playList_unique)*100))

        relevant_items = URM_test[playlist_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(playlist_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            # cumulative_recall += recall(recommended_items, relevant_items)
            # cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    # cumulative_recall /= num_eval
    # cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(cumulative_precision, 0, 0))


# Split input data into tuple
def rowSplit(row_string):
    split = row_string.split(",")
    split[1] = split[1].replace("\n", "")

    split[0] = int(split[0])  # Playlist
    split[1] = int(split[1])  # Song

    return tuple(split)


# PROGRAM MAIN
ICM_matrix = open("./train.csv", 'r')
# print(type(ICM_matrix))

# extract tuples and count how many interaction we have
ICM_tuples = []

print("Fetching data from memory...")
numberInteractions = 0
for line in ICM_matrix:
    numberInteractions += 1
    ICM_tuples.append(rowSplit(line))

print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

# Create one list for each item in the tuple
playList, trackList = zip(*ICM_tuples)

playList = list(playList)
trackList = list(trackList)

# Create some statistics
playList_unique = list(set(playList))
trackList_unique = list(set(trackList))
numPlayLists = len(playList_unique)
numTracks = len(trackList_unique)
# print("Number of playLists: {}\tNumber of tracks: {}".format(numPlayLists, numTracks))
# print("Max ID playlist: {}\tMax Id tracks: {}\n".format(max(playList_unique), max(trackList_unique)))

print("Average tracks per playlist {:.2f}".format(numberInteractions/numPlayLists))
print("Average playlist per track {:.2f}\n".format(numberInteractions/numTracks))
print("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numPlayLists*numTracks))*100))

# Create the CSR Matrix
ICM_matrix = sps.coo_matrix((np.ones(numberInteractions), (playList, trackList)))
print("A {} ICM with {} element".format(ICM_matrix.shape,ICM_matrix.nnz))
ICM_csr = ICM_matrix
ICM_csr.tocsr()

# Getting the more Popular
songPopularity = (ICM_csr > 0).sum(axis=0)
songPopularity = np.sort(np.array(songPopularity).squeeze())
# Build Statistics
tenPercent = int(numTracks/10)
print("Average per-item interactions for the top 10% popular items {:.2f}".format(songPopularity[-tenPercent].mean()))
print("Average per-item interactions for the least 10% popular items {:.2f}".format(songPopularity[:tenPercent].mean()))
print("Number of items with zero interactions {}".format(np.sum(songPopularity == 0)))


# Create a train set
train_test_split = 0.80
train_mask = np.random.choice([True,False], numberInteractions, p=[train_test_split, 1-train_test_split])
userList = np.array(playList)
itemList = np.array(trackList)
URM_train = sps.coo_matrix((np.ones(np.count_nonzero(train_mask == True)), (userList[train_mask], itemList[train_mask]))).tocsr()

# Create a test set
test_mask = np.logical_not(train_mask)
URM_test = sps.coo_matrix((np.ones(np.count_nonzero(test_mask == True)), (userList[test_mask], itemList[test_mask]))).tocsr()

# Build a recommender System
randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)

# Evaluate the algorithm
evaluate_algorithm(URM_test,randomRecommender)


""""
with open('submission.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['playlist_id', 'track_ids'])
    with open('./target_playlists.csv', 'r') as targets:
        for line in targets:
            recommendations = str(randomRecommender.recommend(playList_unique, at=10))
            recommendations = recommendations.replace("[", "")
            recommendations = recommendations.replace("]", "")
            filewriter.writerow([int(line), recommendations])


"""""
