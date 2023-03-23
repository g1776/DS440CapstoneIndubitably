"""
This file contains copied functions for feature engineering. 
This is so we can generate features for the app without having to run the entire feature_engineering notebook.
"""

import pandas as pd
import numpy as np
import pickle
from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from pprint import pprint
from typing import Any
from dataclasses import dataclass


def remove_stopwords_and_lemmatize(tokens) -> list:
    # download stopwords
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    # remove negative words from stopwords
    negative_words = [
        "no",
        "not",
        "nor",
        "neither",
        "never",
        "none",
        "doesnt",
        "couldnt",
        "shouldnt",
        "wouldnt",
        "cant",
        "cannot",
        "wont",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "dont",
        "didnt",
        "neednt",
        "very",
    ]
    for w in negative_words:
        try:
            stop_words.remove(w)
        except KeyError:
            pass

    additional_stopwords = ["airbnb", "austin", "texas", "home", "house"]
    for w in additional_stopwords:
        stop_words.add(w)

    # download lemmatizer
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()

    processed_tokens = []
    for w in tokens:
        if w in stop_words:
            continue
        lemmatized = lemmatizer.lemmatize(w)
        processed_tokens.append(lemmatized)

    return processed_tokens


def preprocess_text(text: str) -> list:
    tokenizer = RegexpTokenizer(r"\w+")

    # remove some specific phrases, using regular expressions
    specific_phrases = [
        r"\(.* hidden by airbnb\)",
    ]

    # lowercase
    text: str = text.lower()

    for phrase in specific_phrases:
        text = re.sub(phrase, "", text)

    # tokenize
    tokens = tokenizer.tokenize(text)

    # remove stopwords and lemmatize
    return remove_stopwords_and_lemmatize(tokens)


def clean_amenities(amenities):
    """Clean the amenities column."""

    cleaned = []

    # basic cleaning
    for amenity in amenities:
        # remove quotes
        amenity = amenity.replace('"', "")
        # remove anything in parentheses or brackets
        amenity = re.sub(r"\(.*\)", "", amenity)
        amenity = re.sub(r"\[.*\]", "", amenity)
        # strip whitespace
        amenity = amenity.strip()
        # lowercase
        amenity = amenity.lower()

        cleaned.append(amenity)

    # split entries with a slash, "and", or "or"
    for to_split_on in ["/", " and ", " or "]:
        cleaned = [amenity.split(to_split_on) for amenity in cleaned]
        cleaned = [item.strip() for sublist in cleaned for item in sublist]

    # remove empty strings
    cleaned = [amenity for amenity in cleaned if amenity != ""]

    return cleaned


@dataclass
class Feature:
    data: pd.DataFrame | np.ndarray
    col: str
    type: str
    models: dict | None
    params: dict | None

    def __str__(self):
        return f"{self.type} ({self.col}) - {self.params}"

    __repr__ = __str__


class DemoClassifier:
    def __init__(self, model_pickle, logging_callback=None):
        self.__logging_callback = logging_callback

        self.print("Loading models...")
        # load the classifier and the word2vec models
        model_pickle = pickle.load(open(model_pickle, "rb"))

        self.__features = model_pickle["combo"]
        self.__clf = model_pickle["model"]

        # self.__w2vs = [feature.models["w2v"] for feature in combo if "w2v" in feature.models]
        # self.__pcas = [feature.models["pca"] for feature in combo if "pca" in feature.models]

        # print(model_pickle)

        # self.__clf = model_pickle["model"]
        # self.__w2v_comments = feature_models_pickle["w2vmodel_comments"]
        # self.__w2v_descriptions = feature_models_pickle["w2vmodel_description"]
        # self.__pca_comments = feature_models_pickle["pca_comments"]
        # self.__pca_descriptions = feature_models_pickle["pca_description"]

        # # load the feature set
        # self.__features: List = model_pickle["best_feature_set"]

        # self.print("Determining features to calculate...")
        # #####################################
        # ## Determine Features to Calculate ##
        # #####################################

        # # figure out with amenities are in the feature set by looking at the names of the features
        # self.__amenities_to_find = []
        # amenity_re = re.compile(r"amenity.*_.*")
        # amenities = set(feature for feature in self.__features if amenity_re.match(feature))
        # for amenity_feature in amenities:
        #     amenity = " ".join(amenity_feature.split("_")[1:])
        #     self.__amenities_to_find.append(
        #         {
        #             "feature": amenity_feature,
        #             "amenity": amenity,
        #         }
        #     )

        # # figure out which embedding features there are in the feature set
        # self.__embeddings_to_generate = []
        # embedding_features = [feature for feature in self.__features if feature.startswith("pca_")]
        # for embedding_feature in embedding_features:
        #     col = embedding_feature.split("_")[1]
        #     n = int(embedding_feature.split("_")[2])
        #     self.__embeddings_to_generate.append(
        #         {"col": col, "feature": embedding_feature, "n": n}
        #     )

        # self.print("Init complete.")

    def __generate_embedding(self, text, w2vmodel):
        # average the word embeddings in the text
        return np.mean([w2vmodel.wv[word] for word in text if word in w2vmodel.wv], axis=0)

    def print(self, msg):
        if self.__logging_callback:
            self.__logging_callback(msg)
        print(msg)

    def predict_all_processed(self, processed: pd.DataFrame):
        processed_features = []

        for row in processed.itertuples():
            row_features: pd.DataFrame = self.predict(
                row.description,
                row.amenities,
                row.comments,
                preprocess=False,
                return_features=True,
            )
            processed_features.append(row_features)

        processed_features = pd.concat(processed_features, axis=0)

        self.print("Predicting...")
        # predict
        predictions = self.__clf.predict(processed_features)
        probabilities = self.__clf.predict_proba(processed_features)
        # align probabilities with labels
        probabilities = [dict(zip(self.__clf.classes_, prob)) for prob in probabilities]

        self.print(f"Prediction: {predictions}")

        return predictions, probabilities

    def predict(self, description, amenities, review, preprocess=True, return_features=False):
        features = {}

        if preprocess:
            self.print("Preprocessing...")
            # preprocess the description
            description = preprocess_text(description)
            # preprocess the review
            review = preprocess_text(review)
            # clean amenities
            amenities = clean_amenities(amenities)

        self.print("Generating embeddings features...")
        print(self.__features)
        for embedding_feature in self.__embeddings_to_generate:
            col = embedding_feature["col"]
            n = embedding_feature["n"]

            if col == "comments":
                embedding = self.__generate_embedding(review, self.__w2v_comments)
                pca_vector = self.__pca_comments.transform(embedding.reshape(1, -1))[0]
            elif col == "description":
                embedding = self.__generate_embedding(description, self.__w2v_descriptions)
                pca_vector = self.__pca_descriptions.transform(embedding.reshape(1, -1))[0]

            print(pca_vector, n)
            embedding_n = pca_vector[n]

            features[embedding_feature["feature"]] = embedding_n

        self.print("Generating ngram features...")
        # generate ngram features
        for ngram_feature in self.__grams_to_find:
            self.print(f"... Calculating Feature {ngram_feature['feature']}")
            features[ngram_feature["feature"]] = ngram_feature["gram"] in description

        self.print("Generating amenity features...")
        # generate amenity features
        for amenity_feature in self.__amenities_to_find:
            self.print(f"... Calculating Feature {amenity_feature['feature']}")
            in_amenities = amenity_feature["amenity"] in amenities
            in_review = amenity_feature["amenity"] in review
            features[amenity_feature["feature"]] = in_amenities and in_review

        # create final features dataframe
        features = pd.DataFrame([features], index=[0])
        # reorder columns to match self.__features
        if "label" in self.__features:
            self.__features.remove("label")
        features = features[self.__features]

        if return_features:
            return features

        self.print("Predicting...")
        # predict
        prediction = self.__clf.predict(features)[0]
        probabilities = self.__clf.predict_proba(features)[0]
        # align probabilities with labels
        probabilities = dict(zip(self.__clf.classes_, probabilities))

        self.print(f"Prediction: {prediction}")

        return prediction, probabilities


if __name__ == "__main__":
    pass

    # def parse_amenities(amenities):
    #     amenities = amenities.replace("{", "").replace("}", "").replace("]", "").replace('"', "")
    #     return amenities.split(",")

    d_clf = DemoClassifier(
        r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\knn_model.pkl"
    )

    # prediction, probabilities = d_clf.predict(
    #     description=(
    #         "Enjoy your stay in a calming home. In the hot Texas sun, cool off with some air"
    #         " conditioning while watching TV, and drink some coffee with the coffee maker. If you"
    #         " want to relax, take a hot tub. This home is dog friendly, so bring your dog along!"
    #     ),
    #     amenities=[
    #         "Internet",
    #         "Kitchen",
    #         "Dogs",
    #         "Air conditioner",
    #         "TV",
    #         "Cable TV",
    #         "coffee maker",
    #         "hot tub",
    #     ],
    #     review=(
    #         "Would not recommend to anyone. This listing was very misleading. The pictures are not"
    #         " as the property looks. The air conditioner is broken."
    #     ),
    # )

    # print(f"Prediction: {prediction}")
    # print(f"Probabilities:")
    # pprint(probabilities)
