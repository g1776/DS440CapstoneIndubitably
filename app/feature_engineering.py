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

        print("Loading training data...")
        # load the training data (used for t-SNE)
        self.__training_data = []
        training_data_geos = ["texas", "florida"]
        for geo in training_data_geos:
            df = pd.read_csv(f"../data/processed/{geo}_processed.csv")
            df["source"] = geo
            self.__training_data.append(df)
        self.__training_data = pd.concat(self.__training_data)

        # load the feature set
        self.__features: List = pickle_loaded["best_feature_set"]

        print("Determining features to calculate...")
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

        # figure out which embedding features there are in the feature set
        self.__embeddings_to_generate = []
        embedding_features = [
            feature for feature in self.__features if feature.startswith("embedding_")
        ]
        for embedding_feature in embedding_features:
            col = embedding_feature.split("_")[1]
            n = int(embedding_feature.split("_")[2])
            self.__embeddings_to_generate.append(
                {"col": col, "feature": embedding_feature, "n": n}
            )

        # figure out which t-SNE features there are in the feature set
        # self.__tsnes_to_generate = []
        # tsnes = [feature for feature in self.__features if feature.startswith("tsne")]
        # for tsne_feature in tsnes:
        #     n_dims = int(tsne_feature[4])
        #     col = tsne_feature.split("_")[1]
        #     dim_number = int(tsne_feature.split("_")[-1])
        #     self.__tsnes_to_generate.append(
        #         {
        #             "feature": tsne_feature,
        #             "n_dims": n_dims,
        #             "dim_number": dim_number,
        #             "col": col,
        #         }
        #     )

        print("Init complete.")

    def __generate_embedding(self, text, w2vmodel):
        # average the word embeddings in the text
        return np.mean([w2vmodel.wv[word] for word in text if word in w2vmodel.wv], axis=0)

    def __generate_training_embeddings(self, col, w2vmodel) -> pd.DataFrame:
        training_embeddings = pd.DataFrame()
        embeddings = np.array(
            [
                self.__generate_embedding(text, w2vmodel)
                for text in self.__training_data[col].to_list()
            ]
        )

        for i in range(embeddings.shape[1]):
            training_embeddings[f"embedding_{i}"] = embeddings[:, i]

        return training_embeddings

    def predict(self, description, amenities, review):
        features = {}

        print("Preprocessing...")
        # preprocess the description
        description = preprocess_text(description)
        # preprocess the review
        review = preprocess_text(review)
        # clean amenities
        amenities = clean_amenities(amenities)

        print("Generating embeddings features...")
        for embedding_feature in self.__embeddings_to_generate:
            col = embedding_feature["col"]
            n = embedding_feature["n"]

            if col == "comments":
                embedding = self.__generate_embedding(review, self.__w2v_comments)
            elif col == "description":
                embedding = self.__generate_embedding(description, self.__w2v_descriptions)

            embedding_n = embedding[n]

            features[embedding_feature["feature"]] = embedding_n

        # print("Generating t-SNE features...")
        # # generate tsne features
        # training_embeddings_description = self.__generate_training_embeddings(
        #     "description", self.__w2v_descriptions
        # )
        # training_embeddings_comments = self.__generate_training_embeddings(
        #     "comments", self.__w2v_comments
        # )
        # tsnes = []
        # for tsne_feature in self.__tsnes_to_generate:
        #     print(f"... Calculating Feature {tsne_feature['feature']}")
        #     if tsne_feature["col"] == "description":
        #         text = description
        #         w2v = self.__w2v_descriptions
        #         training_embeddings = training_embeddings_description
        #     elif tsne_feature["col"] == "comments":
        #         text = review
        #         w2v = self.__w2v_comments
        #         training_embeddings = training_embeddings_comments

        #     # check if a tsne was already created for this column with this number of dimensions
        #     tsne_found = False
        #     for tsne in tsnes:
        #         if tsne["col"] == tsne_feature["col"] and tsne["n_dims"] == tsne_feature["n_dims"]:
        #             # if so, use the already created tsne
        #             prediction_embedding = tsne["tsne_embedding"]
        #             tsne_found = True
        #             break

        #     if not tsne_found:
        #         print("... generating embedding for observation")
        #         embeddings = self.__generate_embedding(text, w2v)

        #         # apply TSNE to the training data + the observation to predict with random_state=0
        #         tsne = TSNE(n_components=tsne_feature["n_dims"], random_state=0)

        #         # add the embedding to the training data embeddings
        #         embeddings_df = pd.DataFrame(
        #             [embeddings],
        #             columns=[f"embedding_{i}" for i in range(embeddings.shape[0])],
        #         )
        #         embeddings_df["to_predict"] = True
        #         embeddings_df = pd.concat([training_embeddings, embeddings_df], axis=0)

        #         # fit the tsne
        #         print("... fitting t-SNE (training data + observation)")
        #         prediction_embedding = tsne.fit_transform(embeddings_df)
        #         # grab the last row, which is the embedding for the observation
        #         prediction_embedding = prediction_embedding[-1, :]
        #         tsnes.append(
        #             {
        #                 "col": tsne_feature["col"],
        #                 "n_dims": tsne_feature["n_dims"],
        #                 "tsne_embedding": prediction_embedding,
        #             }
        #         )

        #     # grab the correct dimension from the tsne embedding
        #     tsne_embedding_n = prediction_embedding[tsne_feature["dim_number"]]

        #     # add the features to the feature set
        #     features[tsne_feature["feature"]] = tsne_embedding_n

        print("Generating ngram features...")
        # generate ngram features
        for ngram_feature in self.__grams_to_find:
            print(f"... Calculating Feature {ngram_feature['feature']}")
            features[ngram_feature["feature"]] = ngram_feature["gram"] in description

        print("Generating amenity features...")
        # generate amenity features
        for amenity_feature in self.__amenities_to_find:
            print(f"... Calculating Feature {amenity_feature['feature']}")
            in_amenities = amenity_feature["amenity"] in amenities
            in_review = amenity_feature["amenity"] in review
            features[amenity_feature["feature"]] = in_amenities and in_review

        # create final features dataframe
        features = pd.DataFrame([features], index=[0])
        # reorder columns to match self.__features
        self.__features.remove("label")
        features = features[self.__features]

        print("Predicting...")
        # predict
        prediction = self.__clf.predict(features)[0]
        probabilities = self.__clf.predict_proba(features)[0]
        # align probabilities with labels
        probabilities = dict(zip(self.__clf.classes_, probabilities))

        return prediction, probabilities


def parse_amenities(amenities):
    amenities = amenities.replace("{", "").replace("}", "").replace("]", "").replace('"', "")
    return amenities.split(",")


d_clf = DemoClassifier(
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\best_clf_texas_florida.pickle",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\w2vmodel_comments_texas_florida_no_tsne.model",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\w2vmodel_description_texas_florida_no_tsne.model",
)

prediction, probabilities = d_clf.predict(
    description=(
        "Enjoy your stay in a calming home. In the hot Texas sun, cool off with some air"
        " conditioning while watching TV, and drink some coffee with the coffee maker. If you want"
        " to relax, take a hot tub. This home is dog friendly, so bring your dog along!"
    ),
    amenities=[
        "Internet",
        "Kitchen",
        "Dogs",
        "Air conditioner",
        "TV",
        "Cable TV",
        "coffee maker",
        "hot tub",
    ],
    review=(
        "Would not recommend to anyone. This listing was very misleading. The pictures are not as the property looks. The air conditioner is broken."
    ),
)

print(f"Prediction: {prediction}")
print(f"Probabilities:")
pprint(probabilities)
