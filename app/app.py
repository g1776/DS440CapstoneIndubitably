import streamlit as st
import pickle

"""
# Airbnb Review Classifier
Is the review of this Airbnb listing indicating that the listing is misleading?
"""


"### Enter the following information about the listing:"

description = st.text_area("Enter a description of the listing")

amenities = st.text_area("Enter the amenities of the listing, separated by commas (e.g. 'wifi, kitchen, parking, etc.")


"### Enter the following information about the review:"

review = st.text_area("Enter a review of the listing")


"### Classify the review:"


classifier_fp = st.text_input("Enter the path to the classifier pickle file", r"C:\Users\grego\Documents\GitHub\DS440CapstoneIndubitably\notebooks\modelling\best_clf_texas_florida.pickle")

if st.button("Classify"):

    with open(classifier_fp, 'rb') as f:
        classifier = pickle.load(f)

    st.write("Classifying...")
    st.write("Classified as: ", classifier.predict(description))