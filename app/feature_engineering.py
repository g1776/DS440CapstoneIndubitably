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


class DemoClassifier:
    def __init__(self, pickle_path, w2v_comments_path, w2v_descriptions_path):
        print("Loading models...")
        # load the classifier and the word2vec models
        pickle_loaded = pickle.load(open(pickle_path, "rb"))
        self.__clf = pickle_loaded["best_clf"]
        self.__w2v_comments = Word2Vec.load(w2v_comments_path)
        self.__w2v_descriptions = Word2Vec.load(w2v_descriptions_path)

        # load the feature set
        self.__features: List = pickle_loaded["best_feature_set"]

        #####################################
        ## Determine Features to Calculate ##
        #####################################

        # figure out which ngrams are in the feature set by looking at the names of the features
        self.__grams_to_find = []
        ngram_re = re.compile(r"[0-9]+gram")
        ngrams = set([feature for feature in self.__features if ngram_re.match(feature)])
        for ngram_feature in ngrams:
            gram = " ".join(ngram_feature.split("_")[1:])
            self.__grams_to_find.append(
                {
                    "feature": ngram_feature,
                    "gram": gram,
                }
            )

        # figure out with amenities are in the feature set by looking at the names of the features
        self.__amenities_to_find = []
        amenity_re = re.compile(r"amenity.*_.*")
        amenities = set(feature for feature in self.__features if amenity_re.match(feature))
        for amenity_feature in amenities:
            amenity = " ".join(amenity_feature.split("_")[1:])
            self.__amenities_to_find.append(
                {
                    "feature": amenity_feature,
                    "amenity": amenity,
                }
            )

        # figure out which t-SNE features there are in the feature set
        self.__tsnes_to_generate = []
        tsnes = [feature for feature in self.__features if feature.startswith("tsne")]
        for tsne_feature in tsnes:
            n_dims = int(tsne_feature[4])
            col = tsne_feature.split("_")[1]
            self.__tsnes_to_generate.append(
                {
                    "feature": tsne_feature,
                    "n_dims": n_dims,
                    "col": col,
                }
            )

        print("Init complete.")

    def __generate_embedding(self, text, w2vmodel):
        # average the word embeddings in the text
        return np.mean([w2vmodel.wv[word] for word in text if word in w2vmodel.wv], axis=0)

    def predict(self, description, amenities, review):
        features = {}

        # preprocess the description
        description = preprocess_text(description)
        # preprocess the review
        review = preprocess_text(review)
        # clean amenities
        amenities = clean_amenities(amenities)

        # generate tsne features
        for tsne_feature in self.__tsnes_to_generate:
            if tsne_feature["col"] == "description":
                text = description
                w2v = self.__w2v_descriptions
            elif tsne_feature["col"] == "comments":
                text = review
                w2v = self.__w2v_comments

            embeddings = self.__generate_embedding(text, w2v)


            # TODO: Fix TSNE, and make sure that there is a feature generated for each dimension

            # # apply TSNE with random_state=0
            # tsne = TSNE(n_components=tsne_feature["n_dims"], random_state=0)
            # tsne_embedding = tsne.fit_transform(embeddings.reshape(-1, 1))

            # # add the features to the feature set
            # features[tsne_feature["feature"]] = tsne_embedding

        # generate ngram features
        for ngram_feature in self.__grams_to_find:
            features[ngram_feature["feature"]] = ngram_feature["gram"] in description

        # generate amenity features
        for amenity_feature in self.__amenities_to_find:
            in_amenities = amenity_feature["amenity"] in amenities
            in_review = amenity_feature["amenity"] in review
            features[amenity_feature["feature"]] = in_amenities and in_review

        features = pd.DataFrame([features], index=[0])

        print(features)


d_clf = DemoClassifier(
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\best_clf_texas_florida.pickle",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\w2vmodel_comments_texas_florida.model",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\w2vmodel_description_texas_florida.model",
)

d_clf.predict(
    "This is a description.",
    ["Pool", "Kitchen", "Air conditioning"],
    "This is a review.",
)