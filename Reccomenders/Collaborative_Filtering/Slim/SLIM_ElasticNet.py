import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
import time, sys
from External_Libraries.Notebooks_utils.evaluation_function import evaluate_algorithm
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
import utils_new as utils
from External_Libraries.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class ElasticNetRecommender(SLIMElasticNetRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train):

        super(ElasticNetRecommender, self).__init__(URM_train)

        self.URM_train = URM_train

    def fit(self, train="-train"):
        self.W_sparse = sps.csr_matrix(sps.load_npz("../../Dataset/similarities/ElasticNet-Sim" + train + ".npz"))
        #print(max(self.W_sparse.data))
        #print(min(self.W_sparse.data))

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, at=10,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        result = super(ElasticNetRecommender, self).recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag,
                                   remove_custom_items_flag, return_scores)
        if return_scores is True:
            return result
        else:
            return result[:at]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_profile = self.URM_train[user_id_array]
        if items_to_compute is None:
            return user_profile.dot(self.W_sparse).toarray()
        else:
            return user_profile.dot(self.W_sparse[items_to_compute]).toarray()

def mainElastic():
    #users = utils.get_target_users("../../../Dataset/target_users.csv")
    URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/old/data_train.npz"))
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    #URM_train, URM_validation = train_test_holdout(URM_train, train_perc=0.9)
    #recommender = SLIMElasticNetRecommender(URM_train)
    recommender = ElasticNetRecommender(URM_train)
    recommender.fit("-train")

#mainElastic()

'''''
with open("../../../../Outputs/elasticNet.csv", 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in users:
        f.write(str(user_id) + ", " + utils.trim(recommender.recommend(user_id)[:10]) + "\n")
'''