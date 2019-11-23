import utils
import numpy as np
from Zeus.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise as train_test_leaveoneout
from Zeus.evaluation_function import evaluate_algorithm
from Zeus.Compute_Similarity_Python import Compute_Similarity_Python


URM_matrix = utils.create_coo("../../dataset/data_train.csv")
ICM_matrix = utils.create_coo("../../refinedDataSet/ICM_sub_class.csv")
ICM_matrix_price = utils.create_coo("../../refinedDataSet/ICM_prices.csv")

ICM_matrix_price = ICM_matrix_price.tocsr()
ICM_matrix = ICM_matrix.tocsr()

URM_train, URM_test = train_test_leaveoneout(URM_matrix, use_validation_set=False)


class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM, ICM_2):
        self.URM = URM
        self.ICM = ICM
        self.ICMv2 = ICM_2
        self.W_sparse_class = 0
        self.W_sparse_price = 0
        self.topPop = [0,1,2,3,4,5,6,7,8,9,10]


    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object_class = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)
        similarity_object_price = Compute_Similarity_Python(self.ICMv2.T, shrink=shrink-25,
                                                            topK=topK+20, normalize=normalize,
                                                            similarity=similarity)

        print("Start computing similarity...")
        self.W_sparse_class = similarity_object_class.compute_similarity()
        self.W_sparse_price = similarity_object_price.compute_similarity()
        print("Done!")

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores_class = user_profile.dot(self.W_sparse_class).toarray().ravel()
        scores_price = user_profile.dot(self.W_sparse_price).toarray().ravel()

        if exclude_seen:
            scores_class = self.filter_seen(user_id, scores_class)
            scores_price = self.filter_seen(user_id, scores_price)

        scores = np.array(list(map(lambda x, y: x+y, scores_class.tolist(), scores_price.tolist())))

        # rank items
        ranking = scores.argsort()[::-1]
        if (scores > 1).sum() > 0:
            print("Item with more then one similarity: " + str((scores > 1).sum()))

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


recommender = ItemCBFKNNRecommender(URM_train, ICM_matrix, ICM_matrix_price)
recommender.fit(shrink=125, topK=80)

users = utils.get_first_column("../../dataset/data_target_users_test.csv")


'''with open("output.csv", 'w') as f:
    for user_id in users:
        recommendations = str(recommender.recommend(user_id, at=10))
        recommendations = recommendations.replace("[", "")
        recommendations = recommendations.replace("]", "")
        f.write(str(user_id) + ", " + recommendations + "\n")
'''
result_dict = evaluate_algorithm(URM_test, recommender, users)

'''
x_tick = [10, 50, 100, 200, 500]
MAP_per_k = []

for topK in x_tick:
    recommender = ItemCBFKNNRecommender(URM_train, ICM_matrix)
    recommender.fit(shrink=0.0, topK=topK)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])


pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('TopK')
pyplot.savefig("topk.png")

x_tick = [0, 10, 50, 100, 200, 500]
MAP_per_shrinkage = []

for shrink in x_tick:
    recommender = ItemCBFKNNRecommender(URM_train, ICM_matrix)
    recommender.fit(shrink=shrink, topK=100)

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_shrinkage.append(result_dict["MAP"])

pyplot.plot(x_tick, MAP_per_shrinkage)
pyplot.ylabel('MAP')
pyplot.xlabel('Shrinkage')
pyplot.savefig("shrink.png")
'''