import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import spatial
from sklearn import neighbors

class content_filtering():
    """Content filtering (provide dense matrices).

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def user_calc(self, user, profile):
        # Calculate user profile for individual user
        user_profile = profile.multiply(user, axis=0).mean(axis=0)
        return user_profile

    def fit(self, item_profile, utility_matrix):
        self.utility_matrix = utility_matrix
        self.user_means = self.utility_matrix.mean(axis=1)
        self.item_profile = item_profile.divide(item_profile.max(axis=0), axis=1)  # Normalize item profile
        self.user_profile = self.utility_matrix.apply(self.user_calc, axis=1, profile=self.item_profile).subtract(self.user_means, axis=0)
        distances, indices = self.top_n_calc(self.user_profile, self.item_profile)
        return distances, indices

    def top_n_calc(self, user_profile, item_profile, n=10):
        self.LSH = neighbors.LSHForest(n_estimators=10, radius=1.0, n_candidates=50, n_neighbors=5)
        self.LSH.fit(item_profile.fillna(0))
        distances, indices = self.LSH.kneighbors(user_profile, n_neighbors=n)
        return distances, indices

    def predict_item(self, item_profile, user):
        user_data = self.user_profile.loc[user, :]
        item = item_profile.fillna(0)
        prediction = (user_data * item).sum() + self.user_means[user]
        return prediction

    ########Following parts deal with filtering out already reviewed items########
    def filter_top_n(self, indices):
        missing_vals = np.argwhere(~np.isnan(self.utility_matrix.values))
        mask_arr = self.perform_mask(indices, missing_vals)
        good_arr = self.set_good_index(mask_arr)
        top_n_df = pd.DataFrame(good_arr, index=self.utility_matrix.index)
        top_n_df.replace(list(range(len(self.utility_matrix.columns))), self.utility_matrix.columns, inplace=True)
        return top_n_df

    def perform_mask(self, indices, missing_vals):
        # Given the indices of top_n, and list of already rated items, get mask
        mask = np.zeros(shape=indices.shape)
        for rating in missing_vals:
            row = indices[rating[0]]
            item_index = np.where(row == rating[1])[0][0]
            mask[rating[0], item_index] = 1
        arr = ma.array(indices, mask=mask)
        return arr

    def set_good_index(self, mask_arr, n=10):
        good = np.zeros(shape=mask_arr.shape)
        for i, user in enumerate(mask_arr):
            add = user[user.mask == False][:n]
            good[i, :add.shape[0]] = add
        return good