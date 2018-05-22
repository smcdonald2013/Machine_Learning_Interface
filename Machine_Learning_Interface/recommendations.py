import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import spatial
from sklearn import neighbors

class recommendation_base():
    """Base class for recommendation models.

    Parameters
    ----------
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def normalize_utility_matrix(self, utility_matrix):
        """ De-means user ratings.

        This helps make sure results aren't impacted by users who give consistently give high/low ratings.

        Parameters
        ----------
        utility_matrix : pd.DataFrame, shape (n_users, n_items)
            Matrix of user ratings of items. Unrated items should have NaN as value in matrix.

        Returns
        -------
        utility_norm : pd.DataFrame, shape (n_users, n_items)
            Utility matrix with ratings subtracted by user's average rating.
        """
        self.user_means = utility_matrix.mean(axis=1)
        utility_norm = utility_matrix.subtract(self.user_means, axis=0).fillna(0)
        return utility_norm

    def top_n_calc(self, lsh_fit_mat, neighbors_mat, n=10):
        """ Performs Locally Sensitive Hashing to group similar users/items.

        Note that lsh_fit_mat and neighbors_mat must have same number of columns.

        Parameters
        ----------
        lsh_fit_mat : pd.DataFrame
            Matrix on which LSH should be performed.
        neighbors_mat : pd.DataFrame
            Matrix on which neighbors are found.
        n : int
            Number of neighbors to return.

        Returns
        -------
        distances : np.array, shape (neighbors_mat.shape[0], n)
            Distances of rows of neighbors_mat to indices.
        indices : np.array, shape (neighbors_mat.shape[0], n)
             Indices of n most similar members of lsh_fit_mat for members of neighbors_mat.
        """
        self.LSH = neighbors.LSHForest(n_estimators=10, radius=1.0, n_candidates=50, n_neighbors=n)
        self.LSH.fit(lsh_fit_mat)
        distances, indices = self.LSH.kneighbors(neighbors_mat, n_neighbors=n)
        return distances, indices

    ########Following parts deal with filtering out already reviewed items########
    def filter_top_n(self, indices):
        """ Given indices of items to recommend to users, filter out those belonging to items each user has already rated.

        Currently only works for content filtering, but in principle should work for collaborative filtering as well.

        Parameters
        ----------
        indices : np.array, shape (n_users, n)
            Indices of n recommended items for each user.

        Returns
        -------
        top_n_df : np.array, shape (n_users, n)
            Names (not indices) of n recommended and not previously rated items for each user.
        """
        rated_indices = np.argwhere(~np.isnan(self.utility_matrix.values))
        mask_arr = self.perform_mask(indices, rated_indices)
        good_arr = self.set_good_index(mask_arr)
        top_n_df = pd.DataFrame(good_arr, index=self.utility_matrix.index)
        top_n_df.replace(list(range(len(self.utility_matrix.columns))), self.utility_matrix.columns, inplace=True)
        return top_n_df

    def perform_mask(self, indices, rated_idx):
        """ Given indices of items to recommend to users, and indices of items users have already rated, create masked array for ratings.

        Parameters
        ----------
        indices : np.array, shape (n_users, n)
            Indices of n recommended items for each user.
        rated_idx : np.array, shape (n_users, n)
            Indices of items rated by each user.

        Returns
        -------
        arr : np masked array, shape (n_users, n)
            Masked points are those in indices matrix corresponding to items user has already rated.
        """
        mask = np.zeros(shape=indices.shape)
        for rating in rated_idx:
            row = indices[rating[0]]
            item_index = np.where(row == rating[1])[0][0]
            mask[rating[0], item_index] = 1
        arr = ma.array(indices, mask=mask)
        return arr

    def set_good_index(self, mask_arr, n=10):
        """ Given masked array, generates matrix of indices excluding masked items.

        Parameters
        ----------
        mask_arr : np masked array, shape (n_users, n)
            Masked points are those in indices matrix corresponding to items user has already rated.
        n : int
            Number of recommendations to retain for each user

        Returns
        -------
        good : np.array, shape (n_users, n)
            Indices of recommended items for each user.
        """
        good = np.zeros(shape=mask_arr.shape)
        for i, user in enumerate(mask_arr):
            add = user[user.mask == False][:n]
            good[i, :add.shape[0]] = add
        return good

class content_filtering(recommendation_base):
    """Content filtering (provide dense matrices).

    Parameters
    ----------
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def user_calc(self, user, profile):
        """ Find average rating given by user to items containing each feature of profile.

        Note that this is not matrix multiplication. The ratings are averaged, not summed as in matrix multiplication.

        Parameters
        ----------
        user : np.array, shape (1, n_items)
            Masked points are those in indices matrix corresponding to items user has already rated.
        profile : np.array, shape (n_items, n_features)
            Profiles of each item.

        Returns
        -------
        user_profile : np.array, shape (1, n_features)
            User preferences (average rating) for each feature
        """
        user_profile = profile.multiply(user, axis=0).mean(axis=0)
        return user_profile

    def fit(self, item_profile, utility_matrix, n=10):
        """ Given item profile and utility matrix, find n most recommended items for each user's tastes.

        Parameters
        ----------
        item_profile : np.array, shape (n_items, n_features)
            Masked points are those in indices matrix corresponding to items user has already rated.
        utility_matrix : np.array, shape (n_users, n_items)
            User ratings of items. Unrated items should have NaN as value.
        n : int
            Number of neighbors to return.

        Returns
        -------
        distances : np.array, shape (n_users, n)
            Distances of users to items given by indices.
        indices : np.array, shape (n_users, n)
             Indices of n most similar items to each user's profile.
        """
        self.utility_matrix = utility_matrix
        self.item_profile = item_profile.divide(item_profile.max(axis=0), axis=1)  # Normalize item profile
        normalize_utility_mat = self.normalize_utility_matrix(utility_matrix)
        self.user_profile = normalize_utility_mat.apply(self.user_calc, axis=1, profile=self.item_profile)
        distances, indices = self.top_n_calc(self.item_profile.fillna(0), self.user_profile, n=n)
        return distances, indices

    def predict_item(self, item_profile, user):
        """ Predict user rating for individual item, given profile.

        A benefit of content filtering is that predictions can be made for items with no ratings, as long as they have a profile.

        Parameters
        ----------
        item_profile : np.array, shape (n_items, n_features)
            Values of features for each item.
        user : str
            Name of user to calculate prediction for (must match value from utility_matrix index.

        Returns
        -------
        prediction : float
            User predicted rating for item.
        """
        user_data = self.user_profile.loc[user, :]
        item = item_profile.fillna(0)
        prediction = (user_data * item).sum() + self.user_means[user]
        return prediction

class collaborative_filtering(recommendation_base):
    """Collbarotaive filtering recommendation models: user-user and item-item similarity.

    Parameters
    ----------
    type:   str, one of 'user', 'item'
        Whether to perform user-user or item-item similarity.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, type='user'):
        self.type = type

    def fit(self, utility_matrix, n=10):
        """ Given  utility matrix, find n most similar users/items to each user/item (depending on whether type=user or type=item).

        Parameters
        ----------
        utility_matrix : np.array, shape (n_users, n_items)
            User ratings of items. Unrated items should have NaN as value.
        n : int
            Number of neighbors to return.

        Returns
        -------
        distances : np.array, shape (n_users, n)
            Distances of every user/item to n most similar users/items
        indices : np.array, shape (n_users, n)
             Indices of n most similar users/items to every user/item
        """
        self.utility_matrix = utility_matrix
        self.utility_norm = self.normalize_utility_matrix(utility_matrix)
        if self.type == 'item':
            utility_pass = self.utility_norm.transpose()
        elif self.type == 'user':
            utility_pass = self.utility_norm
        distances, indices = self.top_n_calc(utility_pass, utility_pass, n=n)
        return distances, indices

    def avg(self, select_id, indices):
        """ Given indices and an id, finds average rating of most similar users/items (depending on type=user or type=item).

        Note that this is the average rating on all users/items. So, for user-user similarity, this will give the average rating of the n
        users most similar to user select_id, for ALL items. For item-item similarity, this will give the average rating given by all users on the
        n items most similar to item select_id.

        Parameters
        ----------
        select_id : int
            Index of user/item to find average rating.
        indices : np.array (n_users/items, n)
            Indices of n most similar users/items for each user/item.

        Returns
        -------
        pred : np.array, shape (n_users/items, 1)
            Average rating of 0 most similar users/items.
        """
        similar_id = indices[select_id, 1:]
        if self.type == 'user':
            pred = self.utility_norm.iloc[similar_id, :].mean(axis=0)
            # user_pred += self.user_means[user_id] #Adding back in user average rating, not needed if all we want is ranking
        elif self.type == 'item':
            pred = self.utility_norm.iloc[:, similar_id].mean(axis=0)
        return pred

    def top_n(self, indices, n=3):
        """ Generates top n recommendations for all users/items, depending on whether type=user or type=item.

        Note that in the case of user-user similarity, this is the typical expected recommendation matrix. For
        item-item similarity, this gives item recommendations for each item- this it is item, and not user, specific.

        Parameters
        ----------
        indices : np.array (n_users/items, n)
            Indices of n most similar users/items for each user/item.
        n : int
            Number of recommendations to return.

        Returns
        -------
        top_n_df : np.array, shape (n_users/items, n)
            Recommended items for each user/item.
        """
        n_select = indices.shape[0]
        if self.type == 'user':
            top_n_df = pd.DataFrame(index=self.utility_norm.index, data=np.ones(shape=[n_select, n]))
        elif self.type == 'item':
            top_n_df = pd.DataFrame(index=self.utility_norm.columns, data=np.ones(shape=[n_select, n]))
        for i in range(n_select):
            pred = self.avg(i, indices)
            top_n = pred.nlargest(n)
            top_n_df.iloc[i, :] = top_n.index
        return top_n_df

    def prediction_single(self, user_id, item_id, indices):
        """ Predicted rating of specific item by specific user.

        For user-user similarity, this is the average rating given to the item by n users most similar to user.
        For item-item similarity, this is the average rating given by the user to the n items most similar to item.

        Parameters
        ----------
        user_id : str
            Name (not index) of user to generate prediction for.
        item_id: str
            Name (not index) of item to generate prediction for.
        indices : np.array (n_users/items, n)
            Indices of n most similar users/items for each user/item.

        Returns
        -------
        pred : float
            Predicted rating of item by user.
        """
        user_idx = np.where(self.utility_matrix.index == user_id)[0][0]
        item_idx = np.where(self.utility_matrix.columns == item_id)[0][0]
        if self.type == 'user':
            similar_id = indices[user_idx, 1:]
            pred = self.utility_norm.iloc[similar_id, item_idx].mean(axis=0)
        elif self.type == 'item':
            similar_id = indices[item_idx, 1:]
            pred = self.utility_norm.iloc[user_idx, similar_id].mean(axis=0)
        pred += self.user_means[user_idx]  # Adding back in user_average rating
        return pred

    ##Item-item similarity specific functions follow
    def top_n_rated_items(self, user, n=10):
        """ Given a user's ratings, return indices of top n rated items.

        Parameters
        ----------
        user : pd.Series (n_items, 1)
            Ratings of all items by specific user.
        n : int
            Number of recommendations to return.

        Returns
        -------
        top_n : np.array (n, 1)
            Indices of top n rated items by user.  
        """
        top_n = np.argsort(user.values)[::-1][:n]
        return top_n

    def top_n_item_user(self, indices, n=10):
        """ Gives top n recommended items for each user in the case of item-item similarity.

        Finds top n rated items for each user, and item most similar to each of those items.

        Parameters
        ----------
        indices : np.array (n_items, n)
            Indices of n most similar items for each item.
        n : int
            Number of recommendations to return.

        Returns
        -------
        top_n_df : pd.DataFrame (n_users, n)
            Top n recommended items for each user.
        """

        top_n_df = self.utility_matrix.apply(self.top_n_rated_items, axis=1, n=10)
        top_n_df.replace(indices[:, 0], indices[:, 1], inplace=True)
        top_n_df.replace(list(range(len(self.utility_matrix.columns))), self.utility_matrix.columns, inplace=True)
        top_n_df.columns = list(range(n))
        return top_n_df