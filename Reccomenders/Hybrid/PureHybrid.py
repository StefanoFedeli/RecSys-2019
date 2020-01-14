from External_Libraries.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

import numpy as np

class HybridReccomender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, Rec_4, Rec_5, Rec_6, Rec_7, Rec_8, Rec_9, Rec_10):
        super(HybridReccomender, self).__init__(URM_train)

        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.Recommender_4 = Rec_4
        self.Recommender_5 = Rec_5
        self.Recommender_6 = Rec_6
        self.Recommender_7 = Rec_7
        self.Recommender_8 = Rec_8
        self.Recommender_9 = Rec_9
        self.Recommender_10 = Rec_10

        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.i = 0
        self.l = 0
        self.norm = False

        self.maxW = 0
        self.minW = 0

    def __str__(self):
        return "alpha={0:.3f}, beta={1:.3f}, gamma={2:.3f}, d={3:.3f}, e={4:.3f}, f={5:.3f}, g={6:.3f}, h={7:.3f} i={8:.3f}, l={9:.3f}, norm:{10}".format(
                                                                       self.alpha, self.beta, self.gamma,
                                                                       self.d, self.e, self.f, self.g, self.h,
                                                                       self.i, self.l, self.norm)

    def fit(self, n=False, alpha=0., beta=0., gamma=0., d=0., e=0., f=0., g=0., h=0., i=0., l=0.):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.i = i
        self.l = l
        self.norm = n

    def normalize(self, scores):
        MaxD = np.amax(scores)
        MinD = np.amin(scores)
        coef = (self.maxW - self.minW) / (MaxD - MinD)
        return scores * coef


    def _compute_item_score(self, user_id_array, items_to_compute=False):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.Recommender_4._compute_item_score(user_id_array)
        item_weights_5 = self.Recommender_5._compute_item_score(user_id_array)
        item_weights_6 = self.Recommender_6._compute_item_score(user_id_array)
        item_weights_7 = self.Recommender_7._compute_item_score(user_id_array)
        item_weights_8 = self.Recommender_8._compute_item_score(user_id_array)
        item_weights_9 = self.Recommender_9._compute_item_score(user_id_array)
        item_weights_10 = self.Recommender_10._compute_item_score(user_id_array)


        self.maxW = max(np.amax(item_weights_1),np.amax(item_weights_2),np.amax(item_weights_3),
                        np.amax(item_weights_4),np.amax(item_weights_5),np.amax(item_weights_6),
                        np.amax(item_weights_7),np.amax(item_weights_8),np.amax(item_weights_9),
                        np.amax(item_weights_10))
        self.minW = min(np.amin(item_weights_1),np.amin(item_weights_2),np.amin(item_weights_3),
                        np.amax(item_weights_4),np.amax(item_weights_5),np.amin(item_weights_6),
                        np.amin(item_weights_7),np.amin(item_weights_8),np.amin(item_weights_9),
                        np.amin(item_weights_10))
        if self.norm:
            item_weights_1 = self.normalize(item_weights_1)
            item_weights_2 = self.normalize(item_weights_2)
            item_weights_3 = self.normalize(item_weights_3)
            item_weights_4 = self.normalize(item_weights_4)
            item_weights_5 = self.normalize(item_weights_5)
            item_weights_6 = self.normalize(item_weights_6)
            item_weights_7 = self.normalize(item_weights_7)
            item_weights_8 = self.normalize(item_weights_8)
            item_weights_9 = self.normalize(item_weights_9)
            item_weights_10 = self.normalize(item_weights_10)

        item_weights = item_weights_1 * self.alpha +\
                       item_weights_2 * self.beta +\
                       item_weights_3 * self.gamma +\
                       item_weights_4 * self.d +\
                       item_weights_5 * self.e +\
                       item_weights_6 * self.f +\
                       item_weights_7 * self.g +\
                       item_weights_8 * self.h + \
                       item_weights_9 * self.i + \
                       item_weights_10 * self.l

        return item_weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, at=10,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        result = super().recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag,
                                   remove_custom_items_flag, return_scores)
        if return_scores is True:
            return result
        else:
            return result[:at]
