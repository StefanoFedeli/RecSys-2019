import utils_new as utils
import numpy as np
import scipy.sparse as sps
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from External_Libraries.Notebooks_utils.evaluation_function import evaluate_algorithm
import matplotlib.pyplot as pyplot
from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout
from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from External_Libraries.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from External_Libraries.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from External_Libraries.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os
from External_Libraries.DataIO import DataIO
from External_Libraries.Base.Recommender_utils import check_matrix, similarityMatrixTopK
from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from External_Libraries.GraphBased.P3alphaRecommender import P3alphaRecommender
from External_Libraries.Recommender_utils import check_matrix, similarityMatrixTopK
from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


URM_all = sps.csr_matrix(sps.load_npz("../../../Dataset/data_all.npz"))
URM_train = sps.csr_matrix(sps.load_npz("../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))

features1 = utils.get_second_column("../../../Dataset/ICM_sub_class.csv")
features2 = utils.get_second_column("../../../Dataset/ICM_price.csv")
features3 = utils.get_second_column("../../../Dataset/ICM_asset.csv")
features = features1 + features2 + features3

items1 = utils.get_first_column("../../../Dataset/ICM_sub_class.csv")
items2 = utils.get_first_column("../../../Dataset/ICM_price.csv")
items3 = utils.get_first_column("../../../Dataset/ICM_asset.csv")
items = items1 + items2 + items3

ones = np.ones(len(features))

n_items = URM_all.shape[1]
n_tags = max(features) + 1

ICM_shape = (n_items, n_tags)
ICM_all = sps.coo_matrix((ones, (items, features)), shape=ICM_shape)
ICM_all = ICM_all.tocsr()

itemKNNCF = ItemKNNCFRecommender(URM_train)
itemKNNCF.fit(shrink=691, topK=830)




P3alpha = P3alphaRecommender(URM_train)
P3alpha.fit()

class ItemKNNSimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__(URM_train)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')


    def fit(self, topK=100, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)
        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()


hybridrecommender = ItemKNNSimilarityHybridRecommender(URM_train, itemKNNCF.W_sparse, P3alpha.W_sparse)
hybridrecommender.fit(alpha = 0.8)

URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)
URM_train, URM_validation = train_test_holdout(URM_train, train_perc = 0.9)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
print(evaluator_validation.evaluateRecommender(hybridrecommender))
hybridrecommender.recommend()




