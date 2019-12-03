from Old import utils
import numpy as np
from sklearn import preprocessing
import scipy.sparse as sps
from Clean.External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from Clean.External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


URM_matrix = utils.create_coo("../../dataset/URM.csv")
URM_matrix = URM_matrix.tocsr()

features1 = utils.get_features("../../refinedDataSet/UCM_age.csv.csv")
features2 = utils.get_features("../../refinedDataSet/UCM_region.csv.csv")
features = features1 + features2

le = preprocessing.LabelEncoder()
le.fit(features)

featuresList = le.transform(features)

entity1 = utils.get_entity("../../refinedDataSet/UCM_age.csv.csv")
entity2 = utils.get_entity("../../refinedDataSet/UCM_region.csv.csv")
entities = entity1 + entity2

ones = np.ones(len(featuresList))

n_users = URM_matrix.shape[0]
n_tags = max(features) + 1

UCM_shape = (n_users, n_tags)
UCM_all = sps.coo_matrix((ones, (entities, featuresList)), shape=UCM_shape)
UCM_all = UCM_all.tocsr()

URM_train, URM_test = train_test_holdout(URM_matrix, train_perc = 0.8)


#da modificare
<class ItemCBFKNNRecommender(object):

    def __init__(self, URM, UCM):
        self.URM = URM
        self.UCM = UCM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


recommender = ItemCBFKNNRecommender(URM_train, UCM_all)
recommender.fit(shrink=200, topK=10)

users = utils.get_first_column("../../dataset/target_users.csv")

with open("output1.csv", 'w') as f:
    for user_id in users:
        recommendations = str(recommender.recommend(user_id, at=10))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(user_id) + ", " + recommendations + "\n")

