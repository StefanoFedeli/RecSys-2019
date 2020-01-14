import utils_new as utils
import numpy as np
import scipy.sparse as sps
import evaluator as evaluate

from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

URM = sps.csr_matrix(sps.load_npz("../../../../Dataset/old/data_train.npz"))

features1 = utils.get_second_column("../../../../Dataset/UCM/UCM_age.csv", seek=13)
features2 = utils.get_second_column("../../../../Dataset/UCM/UCM_region.csv", seek=13)
features = features1 + features2

entity1 = utils.get_first_column("../../../../Dataset/UCM/UCM_age.csv", seek=13)
entity2 = utils.get_first_column("../../../../Dataset/UCM/UCM_region.csv", seek=13)
entities = entity1 + entity2

ones = np.ones(len(features))
UCM_all = sps.coo_matrix((ones, (entities, features)), shape=URM.shape)
UCM_all = UCM_all.tocsr()

UCM = sps.coo_matrix((np.ones(len(features1)), (entity1, features1)))
UCM = UCM.tocsr()


class UserCBFKNNRecommender(BaseUserSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "UserKNNCBRecommender"

    def __init__(self, URM, UCM):
        super().__init__(URM)
        self.URM = URM
        self.UCM = UCM

    def fit(self, topk=50, shrink=100, similarity='cosine', normalize=True):
        similarity_object = Compute_Similarity_Python(self.UCM.T, shrink=shrink,
                                                      topK=topk, normalize=normalize,
                                                      similarity=similarity)
        self.W_sparse = similarity_object.compute_similarity()


sps.save_npz("../../../../Dataset/UCM/UCM_all.npz", UCM_all)

URM_test = sps.csr_matrix(sps.load_npz("../../../../Dataset/old/data_test.npz"))
users = utils.get_target_users("../../../../Dataset/target_users.csv", seek=9)
validator = validate(URM_test, [10])

reccomender = UserCBFKNNRecommender(URM, UCM_all)

similary = ["cosine", "adjusted", "asymmetric", "pearson", "jaccard", "dice", "tversky", "tanimoto"]
for sim in range(7, 3, -1):
    print("++++++ WORKING ON {0} +++++++".format(similary[sim]))
    for i in range(10):
        for k in range(10,1000, 80):
            for s in range(10, 1000, 80):
                for norm in [True, False]:
                    print("SHRINK:{0}, K:{1}, SIMILARITY:{2}, NORM={3}".format(s,k,similary[sim],norm))
                    reccomender.fit(k,s,similary[sim],norm)
                    print(evaluate.evaluate(users, reccomender, URM_test, 10)["MAP"])
                    results = validator.evaluateRecommender(reccomender)
                    print(results[0][10]["MAP"])
                    print("\n")
