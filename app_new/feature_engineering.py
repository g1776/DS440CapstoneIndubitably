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

        self.__combo = model_pickle["combo"]
        self.__clf = model_pickle["model"]

        self.__amenity_features: List[Feature] = []
        self.__embedding_features: List[Feature] = []
        self.__pca_features: List[Feature] = []

        for feature in self.__combo:
            if feature.type == "amenities":
                self.__amenity_features.append(feature)
            elif feature.type == "embedding":
                self.__embedding_features.append(feature)
            elif feature.type == "pca":
                self.__pca_features.append(feature)

    def __generate_embedding(self, text, w2vmodel):
        # average the word embeddings in the text
        return np.mean([w2vmodel.wv[word] for word in text if word in w2vmodel.wv], axis=0)

    def print(self, msg):
        if self.__logging_callback:
            self.__logging_callback(msg)
        print(msg)

    def predict_all_processed(self, processed: pd.DataFrame):
        self.print(f"Predicting on {len(processed)} rows...")

        processed_features = []

        num_rows = len(processed)

        for row in processed.itertuples():
            self.print(f"Processing row {int(row.Index) + 1} of {num_rows}...")

            row_features: pd.DataFrame = self.predict(
                row.description,
                row.amenities,
                row.comments,
                preprocess=False,
                return_features=True,
            )
            processed_features.append(row_features)

        processed_features = pd.concat(processed_features, axis=0)
        processed_features.index = processed.index

        self.print("Processed features:")
        self.print(processed_features.head())

        self.print("Predicting...")
        # predict
        predictions = self.__clf.predict(processed_features)
        probabilities = self.__clf.predict_proba(processed_features)
        # align probabilities with labels
        probabilities = [dict(zip(self.__clf.classes_, prob)) for prob in probabilities]

        self.print(f"Prediction: {predictions}")

        return predictions, probabilities

    def predict(
        self,
        description,
        amenities,
        review,
        preprocess=True,
        return_features=False,
    ):
        features = {}

        if preprocess:
            self.print("Preprocessing...")
            # preprocess the description
            description = preprocess_text(description)
            # preprocess the review
            review = preprocess_text(review)
            # clean amenities
            amenities = clean_amenities(amenities)

        self.print("Generating amenities features...")
        for amenity_feature in self.__amenity_features:
            for amenity_column in amenity_feature.params["amenities"]:
                amenity = amenity_column.split("_")[1]

                features[amenity_column] = int(amenity in amenities)

        self.print("Generating embedding features...")
        for embedding_feature in self.__embedding_features:
            col = embedding_feature.col
            w2v = embedding_feature.models["w2v"]

            if col == "comments":
                embedding = self.__generate_embedding(review, w2v)
            elif col == "description":
                embedding = self.__generate_embedding(description, w2v)

            for i in range(embedding.shape[1]):
                features[f"embedding_{col}_{i}"] = embedding[:, i]

        self.print("Generating pca features...")
        for pca_feature in self.__pca_features:
            col = pca_feature.col
            pca = pca_feature.models["pca"]
            w2v = pca_feature.models["w2v"]

            if "comments" in col:
                embedding = self.__generate_embedding(review, w2v)
            elif "description" in col:
                embedding = self.__generate_embedding(description, w2v)

            embeddings = np.array([embedding])

            # see if we need to include amenities in the data to be transformed

            if "amenities" in col:
                self.print("...with amenities")

                raise NotImplementedError(
                    "Cannot generate PCA with amenities feature. You knew this was coming."
                )

                # TODO: Add this when we need it.
                pass

                # self.print("...with amenities")

                # amenities_df = pd.DataFrame()
                # for amenity_column in amenity_feature.params["amenities"]:
                #     amenity = amenity_column.split("_")[1]
                #     amenities_df[amenity_column] = int(amenity in amenities)

                # to_transform = np.concatenate((embedding, amenities), axis=1)
                # to_transform_df = pd.DataFrame(to_transform).T
                # to_transform_df.columns = [
                #     f"embedding_{col}_{i}" for i in range(to_transform_df.shape[1])
                # ]

            else:
                self.print("...without amenities")

                # add embeddings to features dataframe
                columns = [f"embedding_{col}_{i}" for i in range(embeddings.shape[1])]
                data = [embeddings[:, i] for i in range(embeddings.shape[1])]
                to_transform = pd.concat(
                    [pd.Series(d, name=c) for d, c in zip(data, columns)], axis=1
                )

            pca_vector = pca.transform(to_transform)[0]

            n_components = pca_feature.params["n_components"]
            for i in range(pca_vector.shape[0]):
                features[f"pca_{n_components}D_{col}_{i}"] = pca_vector[i]

        # create final features dataframe
        features = pd.DataFrame([features], index=[0])

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

    d_clf = DemoClassifier(
        r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\knn_model.pkl"
    )

    # prediction, probabilities =
    d_clf.predict(
        description=(
            "Enjoy your stay in a calming home. In the hot Texas sun, cool off with some air"
            " conditioning while watching TV, and drink some coffee with the coffee maker. If you"
            " want to relax, take a dip in the pool. This home is dog friendly, so bring your dog"
            " along!"
        ),
        amenities=[
            "Internet",
            "Kitchen",
            "Dogs",
            "Air conditioner",
            "TV",
            "Cable TV",
            "coffee maker",
            "pool",
        ],
        review=(
            "Would not recommend to anyone. This listing was very misleading. The pictures are not"
            " as the property looks. The air conditioner is broken, and the pool is disgusting."
        ),
    )

    # print(f"Prediction: {prediction}")
    # print(f"Probabilities:")
    # pprint(probabilities)
