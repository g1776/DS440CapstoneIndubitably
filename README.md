# DS 440 Capstone

**Team**: Indubitably

**Members**: Gregory Glatzer, Charlie Lu

**Project**: Airbnb

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/768px-Airbnb_Logo_B%C3%A9lo.svg.png?20140813142239" width="30%" alt="Airbnb"/>

## Repo Content

### Notebooks

Here we will store any Jupyter notebooks for data analysis, preprocessing, etc.

### Scripts

If we want any shared functionality or anything that makes sense to go in a _.py_ file, then put that here.

### Data

Data will be stored here, but not commited (just so its easy for us to run each other's code without modifying paths).

The folders in data are as follows:

-   `raw` - raw data from Airbnb. Also contains <GEO>\_raw_processed.csv, which is processed raw data, which was only used for EDA.
-   `filtered` - subsets of the raw review data that we will label.
-   `labels` - the labels for the filtered data. (can be joined with the filtered data using the `id` column, corresponding to the `listing_id` in the raw data)
-   `sentiment` - the sentiment scores for the raw data. Used to generate the filtered subsets.
-   `processed` - the processed data, ready to be used for training, testing, and validation. Includes labels.

## Reproducibility Steps

1. Clone the repo

2. Install the requirements

```bash
pip install -r requirements.txt
```

3. Make sure you have the folders listed above in the `data` directory. If not, create them.

4. Extract the raw data and put it in the `data/raw` folder. This requires two steps:

    1. Export the listings worksheet from the Excel files, and save it using the naming convention `<GEO>_listings.csv`. For example, `data/raw/texas_listings.csv`.
    2. Place the `.xlsx` files in the `data/raw` folder, and run the `notebooks/data processing/load data.ipynb` notebook. This will extract the reviews from the Excel files and save them in the `data/raw` folder. There is preprocessing needed to correctly load the reviews, so this notebook is necessary.

5. Generate the sentiment scores used for subsetting the data for labelling. You can do this by running the `notebooks/data processing/generate_sentiment.ipynb` notebook. This will save the sentiment scores in the `data/sentiment` folder. This notebook must be run three times, changing the `GEO` variable in the notebook each time. The three values for `GEO` are `texas`, `california`, and `florida`. **NOTE**: This notebook will take hours to run for each geo. We will provide the results of this step for reproducibility.

6. Generate the filtered data for labelling. You can do this by running the `notebooks/data processing/generate_subset.ipynb` notebook. This will save the filtered data in the `data/filtered` folder. You only need to run this once.

7. Label the data. This is done manually, however we have a notebook to help with the task, using the data annotation library `tortus`. The tool we wrote to help us read the reviews and label them is in `notebooks/data processing/label_subsets.ipynb`.

8. Once you have labels, you can generate the processed data. This is where stopword removal, joining labels with the data, etc. happens. This is done by running the `notebooks/data processing/generate_processed.ipynb` notebook. This will save the processed data in the `data/processed` folder. This will be run for each geography, changing the `GEO` variable in the notebook each time.

9. You are now ready to fo feature engineering and modelling. Our entire feature engineering process and generation of our final features for training models is in `notebooks/modelling/feature_engineering.ipynb`. This notebook will generate the final features for each geography, and save them in the `data/processed` folder as `features_<GEO>.csv`. If multiple GEOs are provided it will created the combined features file with both geos in the name. For example, for our training data, we ran this notebook with `GEO = ['texas', 'florida']`, and it saved the features in `data/processed/features_texas_florida.csv`.

10. Modelling! the `notebooks/modelling/models.ipynb` notebook contains the code for generating models and trying different feature combinations. It ends with a confusion matrix.
