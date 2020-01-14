import numpy as np
import scipy.sparse as sps
import utils_new as util
import random
import External_Libraries.Recommender_utils as mauri

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.Base.BaseRecommender import BaseRecommender as BaseRecommender

def normalize(URM_t, minI, maxI):
    MaxD = max(URM_t.data)
    MinD = min(URM_t.data)
    URM = URM_t.copy()
    print(minI,maxI)
    for i in range(URM.nnz):
        URM.data[i] = URM.data[i]*(maxI-minI)/(MaxD-MinD)
    return URM


class ReccomenderSslim(BaseRecommender):

    RECOMMENDER_NAME = "S-SLIMElasticNetRecommender"

    def __init__(self, URM):
        self.URM = URM
        self.W_sparse = np.array([0])

        super().__init__(URM)

    def fit(self, train=""):
        # to train use train="-train"
        URM_1 = sps.csr_matrix(sps.load_npz("../../Dataset/similarities/CB-Sim" + train + ".npz"))
        URM_2 = sps.csr_matrix(sps.load_npz("../../Dataset/similarities/Col-Sim" + train + ".npz"))
        URM_3 = sps.csr_matrix(sps.load_npz("../../Dataset/similarities/Slim-Sim" + train + ".npz"))
        URM_4 = normalize(URM_3, min(min(URM_1.data), min(URM_2.data)), max(max(URM_1.data), max(URM_2.data)))
        self.W_sparse = mauri.similarityMatrixTopK(0.31*URM_1 + 1.82*URM_2 + 0.76*URM_4, k=25)
        print(max(self.W_sparse.data))
        print(min(self.W_sparse.data))

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, at=10,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        result = super().recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag,
                                   remove_custom_items_flag, return_scores)
        if return_scores is True:
            return result
        else:
            return result[:at]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_profile = self.URM[user_id_array]
        if items_to_compute is None:
            return user_profile.dot(self.W_sparse).toarray()
        else:
            return user_profile.dot(self.W_sparse[items_to_compute]).toarray()


def main_sslim():
    URM = sps.csr_matrix(sps.load_npz("../../Dataset/URM/data_all.npz"))
    URM_test = sps.csr_matrix(sps.load_npz("../../Dataset/URM/data_test.npz"))

    URM_1 = sps.csr_matrix(sps.load_npz("../../Dataset/old/similarities/CB-Sim.npz"))
    URM_2 = sps.csr_matrix(sps.load_npz("../../Dataset/old/similarities/Col-Sim.npz"))
    URM_3 = sps.csr_matrix(sps.load_npz("../../Dataset/old/similarities/Slim-Sim.npz"))
    URM_4 = normalize(URM_3, min(min(URM_1.data), min(URM_2.data)), max(max(URM_1.data), max(URM_2.data)))
    mauri_recsys = ReccomenderSslim(URM)
    validator = validate(URM_test, [10])
    targetUsers = util.get_target_users("../../Dataset/target_users.csv", seek=8)
    #similarity_matrix = mauri.similarityMatrixTopK(0.31*URM_1 + 1.82*URM_2 + 0.76*URM_4, k=25)
    mauri_recsys.fit(train="-train")


""" WHAT IS NEED TO TUNE
for i in range(1):
    j = random.uniform(0.2, 1.9)
    z = random.uniform(0.2, 1.9)
    a = random.uniform(0.2, min(j,z))
    j = 1.82
    z = 0.76
    a = 0.31
    for norm in [True]:
        for k in [25]:
            if norm:
                similarity_matrix = mauri.similarityMatrixTopK(a*URM_1 + j*URM_2 + z*URM_4, k=k)
            else:
                similarity_matrix = mauri.similarityMatrixTopK(a * URM_1 + j * URM_2 + z * URM_3, k=k)
            print("NORM:{0}, K:{1}, ContentBased:{2:.2f}, Collaborative:{3:.2f}, Slim:{4:.2f}".format(norm,k,a,j,z))
            mauri_recsys.fit(similarity_matrix)
            #print(evaluate.evaluate(targetUsers, mauri_recsys, URM_test, 10)["MAP"])
            #results = validator.evaluateRecommender(mauri_recsys)
            #print(results[0][10]["MAP"])
            print("\n")
            with open("../../../Outputs/Sslim.csv", 'w') as f:
                f.write("user_id,item_list\n")
                for user_id in targetUsers:
                    #print(user_id)
                    f.write(str(user_id) + "," + util.trim(np.array(mauri_recsys.recommend(user_id))) + "\n")

# NNORM:True, K:25, ContentBased:0.31, Collaborative:1.82, Slim:0.76 @ 0.0243
"""

#main_sslim()