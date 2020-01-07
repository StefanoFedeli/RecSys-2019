

from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class TopPopHybridReccomender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3):
        super(TopPopHybridReccomender, self).__init__(URM_train)

        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3

    def fit(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.gamma

        return item_weights