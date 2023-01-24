import pandas as pd


def semi_supervised_learning(
    model,
    labelled_training_data: pd.DataFrame,
    unlabelled_training_data: pd.DataFrame,
    iters: int = 10,
    pcnt: float = 0.1,
):
    """
    Semi-supervised learning is a machine learning method that combines labelled and unlabelled data to train a model.
    The model is trained on the labelled data and then used to make predictions on the unlabelled data.
    The predictions are then used to train the model on the unlabelled data.
    This process is repeated until the model converges.

    :param model: The model to train.
    :param labelled_training_data: The labelled training data.
    :param unlabelled_training_data: The unlabelled training data.
    :return: The trained model.
    """

    for i in range(iters):

        # train the model on the labelled data
        model.fit(labelled_training_data)

        if unlabelled_training_data.empty:
            print(
                "No more unlabelled data to train on. Stopping training on iteration:"
                f" {i}/{iters}."
            )
            break

        # choose a random pcnt% of the unlabelled data to label
        unlabelled_sample = unlabelled_training_data.sample(frac=pcnt)

        # Error: couldn't load spacey model for any of languages: en
        # Solution: https://stackoverflow.com/questions/56963293/couldnt-load-spacey-model-for-any-of-languages-en

        # make predictions on the unlabelled sample
        predictions = model.predict(unlabelled_sample)

        # add the predictions to the unlabelled sample
        labelled_sample = unlabelled_sample.copy()
        labelled_sample["label"] = predictions

        # update the labelled and unlabelled data
        labelled_training_data = labelled_training_data.append(labelled_sample)
        unlabelled_training_data = unlabelled_training_data.drop(predictions.index)

    return model
