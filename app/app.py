import streamlit as st
import pandas as pd
from feature_engineering import DemoClassifier
import plotly.express as px

# load training data to get list of amenities
florida = pd.read_csv(r"..\data\processed\florida_processed.csv")
texas = pd.read_csv(r"..\data\processed\texas_processed.csv")
training_data = pd.concat([florida, texas], ignore_index=True)
amenities_options = training_data["amenities"].apply(eval).explode().unique()
# remove { and } from amenities
amenities_options = [amenity.replace("{", "").replace("}", "") for amenity in amenities_options]

"""
# Airbnb Review Classifier
Is the review of this Airbnb listing indicating that the listing is misleading?
"""


"### Enter the following information about the listing:"

description = st.text_area("Enter a description of the listing")

amenities = st.multiselect(
    "Choose your amenities",
    amenities_options,
)


"### Enter the following information about the review:"

review = st.text_area("Enter a review of the listing")


"### Classify the review:"


classifier_fp = st.text_input(
    "Enter the path to the classifier pickle file",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\best_clf_texas_florida_pca.pickle",
)
feature_models_fp = st.text_input(
    "Enter the path to the feature models pickle file",
    r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\models\feature_models_texas_florida_pca.pickle",
)

classify = st.button("Classify")

prediction_st = st.empty()
probabilities_st = st.empty()

log_title = st.empty()
log = st.empty()

if classify:
    log_title.write("Log:")
    clf = DemoClassifier(
        classifier_fp,
        feature_models_fp,
        logging_callback=lambda msg: log.write(msg),
    )

    prediction, probabilities = clf.predict(description, amenities, review)

    prediction_st.write(f"Prediction: {prediction}")

    # plot probabilities dictionary as a bar chart
    probabilities_df = pd.DataFrame.from_dict(
        probabilities, orient="index", columns=["Probability"]
    )
    probabilities_df = probabilities_df.reset_index().rename(columns={"index": "Label"})
    fig = px.pie(probabilities_df, values="Probability", names="Label")
    probabilities_st.plotly_chart(fig)
