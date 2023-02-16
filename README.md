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

-   `raw` - raw data from Airbnb. Also contains <GEO>_raw_processed.csv, which is processed raw data, which was only used for EDA.
-   `filtered` - subsets of the raw review data that we will label.
-   `labels` - the labels for the filtered data. (can be joined with the filtered data using the `id` column, corresponding to the `listing_id` in the raw data)
-   `sentiment` - the sentiment scores for the raw data. Used to generate the filtered subsets.
-   `processed` - the processed data, ready to be used for training, testing, and validation. Includes labels.
