import scipy.sparse as sps
import utils_new as utils
import evaluator as evaluate
import random

from External_Libraries.Evaluation.Evaluator import EvaluatorHoldout as validate
from External_Libraries.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

URM = sps.load_npz("../../../Dataset/data_train.npz")
URM_test = sps.csr_matrix(sps.load_npz("../../../Dataset/data_test.npz"))
users = utils.get_target_users("../../../Dataset/target_users.csv", seek=8)
validator = validate(URM_test, [10])

recommender = MF_MSE_PyTorch(URM)

for ep in [30]:
    for batch in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        for i in range(10, 1000, 100):
            for lr in [1e-6]:
                recommender.fit(ep,batch,i,lr)
                print("FACTOR:{0}, BATCH:{1}, LR:{2}, EPOCHS:{3}".format(
                    i, batch, lr, ep))
                print(evaluate.evaluate(users, recommender, URM_test, 10)["MAP"])
                #results = validator.evaluateRecommender(recommender)
                #print(results[0][10]["MAP"])
                print("\n")