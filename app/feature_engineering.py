"""
This file contains copied functions for feature engineering. 
This is so we can generate features for the app without having to run the entire feature_engineering notebook.
"""

import pandas as pd
import pickle
from typing import List
import re


# def get_consecutive_ngrams(review, n) -> list:
#     """Helper function to get ngrams from a review.

#     Args:
#         review (str): The review to get ngrams from.
#         n (int): The number of ngrams to get.

#     Returns:
#         list: the list of ngrams, joined by underscores.
#     """
#     if isinstance(review, str):
#         review = review.split("")

#     return ["_".join(review[i : i + n]) for i in range(len(review) - n - 1)]


# def amenities_features(
#     to_predict: pd.DataFrame, corr_thresh=None, prefix="amenities_"
# ) -> pd.DataFrame:
#     amenities_features_df = pd.DataFrame()
#     # 1. Find the amenities in the listings
#     for index, row in to_predict.iterrows():
#         # 1. Find the amenities in the listings
#         amenities = row.amenities

#         # Add one row to the features dataframe using pd.concat
#         amenities_features_df = pd.concat(
#             [amenities_features_df, pd.DataFrame(columns=amenities_features_df.columns)]
#         )

#         # 2. For each amenity, see if it is present in the review
#         for amenity in amenities:
#             if amenity in row.comments:
#                 # 3. If it is present, add 1 for that feature
#                 if amenity in amenities_features_df.columns:
#                     amenities_features_df.loc[index, amenity] = 1
#                 #   If the amenity does not already exist from another review, add it to the features dataframe
#                 else:
#                     amenities_features_df.loc[index, amenity] = 0

#     # 4. fill missing values with 0
#     amenities_features_df = amenities_features_df.fillna(0)

#     # 6. Only keep amenities features that have a correlation with the label above a certain threshold
#     amenities_features_df = corr_filter(amenities_features_df, corr_thresh=corr_thresh)

#     # prefix features
#     amenities_features_df = amenities_features_df.add_prefix(prefix)

#     # 5. Add the features to the greater features dataframe
#     features = pd.concat([features, amenities_features_df], axis=1)

#     return features


# def series_to_ngrams(series: pd.Series, N):
#     n_grams = series.apply(lambda x: get_consecutive_ngrams(x, N))
#     return n_grams.explode()


# def ngrams_features(features, df, n, prefix="ngrams_", corr_thresh=0.05):
#     """Add n-gram features to the features dataframe."""

#     # one-hot encode ngrams
#     df["ngrams"] = df.comments.apply(lambda x: set(get_consecutive_ngrams(x, 3)))

#     subset = df[df.label.isin(["mbad", "mgood"])]

#     # get set of ngrams
#     ng_set = set(series_to_ngrams(subset.comments, n).to_list())

#     # one-hot encode ngrams
#     ngram_features = {}
#     for ngram in ng_set:
#         ngram_features[prefix + ngram] = df.ngrams.apply(
#             lambda ngrams: 1 if ngram in ngrams else 0
#         )
#     ngram_df = pd.DataFrame(ngram_features)

#     # filter features on correlation with label
#     ngrams_df = corr_filter(ngram_df, corr_thresh=corr_thresh)

#     # add ngram features to features dataframe
#     features = pd.concat([features, ngrams_df], axis=1)

#     return features


class DemoClassifier:
    def __init__(self, pickle_path):
        pickle_loaded = pickle.load(open(pickle_path, "rb"))
        self.__clf = pickle_loaded["best_clf"]
        self.__features: List = pickle_loaded["best_feature_set"]

        # figure out which ngrams are in the feature set by looking at the names of the features
        self.__grams_to_find = []
        ngram_re = re.compile(r"[0-9]+gram")
        ngrams = set([feature for feature in self.__features if ngram_re.match(feature)])
        for ngram_feature in ngrams:
            gram = " ".join(ngram_feature.split("_")[1:])
            self.__grams_to_find.add(
                {
                    "feature": ngram_feature,
                    "gram": gram,
                }
            )
    
    def preprocess_pipeline(self, description, amenities, review):
        pass

    def predict(self, description, amenities, review):
        pass


DemoClassifier(
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\notebooks\modelling\best_clf_texas_florida.pickle"
)
