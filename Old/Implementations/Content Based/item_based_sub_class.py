import scipy.sparse as sps
import numpy as np
import random as rand
from Old import utils as util


# MY RECOMMENDED SYSTEM
class RandomRecommender(object):

    similarity_matrix = {}
    mapping = {}

    def fit(self):
        ICM_matrix = open("../dataset/data_ICM_sub_class.dataset", 'r')
        ICM_matrix.seek(14)

        ICM_tuples = []

        print("Fetching data from memory...")
        numberInteractions = 0
        for line in ICM_matrix:
            numberInteractions += 1
            ICM_tuples.append(rowSplit(line))

        print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

        itemList, featuresList, interactionList = zip(*ICM_tuples)
        itemList = list(itemList)
        featuresList = list(featuresList)


        for item in itemList:
            counter = itemList.index(item);
            if (featuresList[counter] not in self.similarity_matrix):
                self.similarity_matrix[featuresList[counter]] = []
            self.similarity_matrix[featuresList[counter]].append(item)

            self.mapping[item]=featuresList[counter]

    def recommend(self, user_id, liked_items, at=10):
        recommended_items = []
        #print(liked_items)
        for item in liked_items:
            recommended_items.append(self.similarity_matrix[self.mapping.get(item)])
        if(len(recommended_items)>0):
            np.concatenate(recommended_items)
            rand.shuffle(recommended_items[0])
            while(len(recommended_items[0])<10):
                recommended_items[0].append(rand.randint(0, max(itemList)))
            return recommended_items[0][:10]

# METRICS: Precision
def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score

# EVALUATION
def evaluate_algorithm(URM_test, URM_csr, recommender_object, at=10):
    cumulative_precision = 0.0
    # cumulative_recall = 0.0
    # cumulative_MAP = 0.0
    num_eval = 0

    target = open("../dataset/data_target_users_test.dataset", 'r')
    target.seek(9)

    for line in target:
        line = int(line)

        relevant_items = URM_test[line].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(line, URM_csr.getrow(line).toarray().nonzero()[1], at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            # cumulative_recall += recall(recommended_items, relevant_items)
            # cumulative_MAP += MAP(recommended_items, relevant_items)
        print(relevant_items)

    cumulative_precision /= num_eval
    # cumulative_recall /= num_eval
    # cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(cumulative_precision, 0, 0))


###################
# ./MAIN PROGRAM  #
###################
userList, itemList, interactionList = util.create_coo("../dataset/URM.csv", 14)
URM_matrix = sps.coo_matrix(interactionList,(userList,itemList))
print("A {} URM with {} element".format(URM_matrix.shape, URM_matrix.nnz))
URM_csr = URM_matrix.tocsr()

URM_matrix_test = util.create_test_matrix("../dataset/URM.csv", 14)

# Create intelligence
randomRecommender = RandomRecommender()
randomRecommender.fit()
liked_items = URM_csr.getrow(0).toarray().nonzero()
print(randomRecommender.recommend(0, liked_items[1]))

# Evaluate the algorithm
evaluate_algorithm(URM_matrix_test, URM_csr, randomRecommender)







'''
#MAIN
URM_matrix = open("../../dataset/data_ICM_sub_class.dataset", 'r')
URM_matrix.seek(14)

URM_tuples = []

print("Fetching data from memory...")
numberInteractions = 0
for line in URM_matrix:
    numberInteractions += 1
    URM_tuples.append(rowSplit(line))

print("Done! {} tuples (interactions) ingested\n".format(numberInteractions))

userList, itemList, interactionList = zip(*URM_tuples)

userList = list(userList)
itemList = list(itemList)
interactionList = list(interactionList)

userList_unique = list(set(userList))
itemList_unique = list(set(itemList))
numUsers = len(userList_unique)
numItems = len(itemList_unique)

print("Average items per user {:.2f}".format(numberInteractions/numUsers))
print("Average user per item {:.2f}\n".format(numberInteractions/numItems))
print("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numUsers*numItems))*100))


URM_matrix = sps.coo_matrix((interactionList, (userList, itemList)))
print("A {} URM with {} element".format(URM_matrix.shape,URM_matrix.nnz))
URM_csr = URM_matrix
URM_csr.tocsr()
'''