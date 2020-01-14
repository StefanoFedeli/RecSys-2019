import utils_new as utils
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as pyplot
import evaluator as evaluate
import os

from External_Libraries.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from External_Libraries.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from External_Libraries.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from External_Libraries.Notebooks_utils.data_splitter import train_test_holdout
from External_Libraries.Base.Evaluation.Evaluator import EvaluatorHoldout

from External_Libraries.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Real, Integer, Categorical
from External_Libraries.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

URM_all = sps.csr_matrix(sps.load_npz("../../../../Dataset/old/data_all.npz"))
"""
URM_train = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_train.npz"))
URM_test = sps.csr_matrix(sps.load_npz("../../../../../Dataset/data_test.npz"))
"""
ICM_all = sps.csr_matrix(sps.load_npz("../../../../Dataset/ICM/ICM_all.npz"))
URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)
URM_train, URM_validation = train_test_holdout(URM_train, train_perc = 0.9)


itemKNNCF = ItemKNNCFRecommender(URM_train)
itemKNNCF.fit(shrink=25, topK=10, similarity="jaccard", feature_weighting="TF-IDF", normalize=False)

itemKNNCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
itemKNNCBF.fit(shrink=120, topK=5, similarity="asymmetric", feature_weighting="none", normalize=True)


W_sparse_CF = itemKNNCF.W_sparse
W_sparse_CBF = itemKNNCBF.W_sparse

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5])

recommender_class = CFW_D_Similarity_Linalg

parameterSearch = SearchBayesianSkopt(recommender_class,
                                 evaluator_validation=evaluator_validation,
                                 evaluator_test=evaluator_test)


hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["topK"] = Integer(5, 100)
hyperparameters_range_dictionary["add_zeros_quota"] = Real(low = 0, high = 1, prior = 'uniform')
hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_all, W_sparse_CF],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)


output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 25
metric_to_optimize = "MAP"


# Clone data structure to perform the fitting with the best hyperparameters on train + validation data
recommender_input_args_last_test = recommender_input_args.copy()
recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation


parameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       parameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = int(n_cases/3),
                       save_model = "no",
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender_class.RECOMMENDER_NAME,
                       metric_to_optimize = metric_to_optimize
                      )

