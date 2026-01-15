from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from heuristic import HeuristicPredictor
# # from knn import KNNPredictor

from apriori import AprioriPredictor


class Metric:
    def __init__(self):
        self.nan = 0
        self.correct = 0
        self.wrong = 0

    def __str__(self):
        return f"NaN: {self.nan}; correct: {self.correct}, wrong: {self.wrong}; ratio: {self.correct / max(1, (self.correct + self.wrong))}"


def clean_data():
    df = pd.read_csv("listings.csv")
    important_columns = set(
        [
            "neighbourhood",
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds",
        ]
    )
    to_drop = set(df.columns).difference(important_columns)
    df = df.drop(to_drop, axis=1)
    pickle.dump(df, open("data.pkl", "wb"))


def split_data(df, train_size=0.7):
    train_set, test_set = train_test_split(df, test_size=(1 - train_size))
    pickle.dump(train_set, open("train_set.pkl", "wb+"))
    pickle.dump(test_set, open("test_set.pkl", "wb+"))


def train():
    train_set = pickle.load(open("train_set.pkl", "rb"))
    hPredictor = HeuristicPredictor(train_set)
    aPredictor = AprioriPredictor(train_set, 0.05, 0.05)

    return hPredictor, aPredictor


def validate(predictor):
    test_set = pickle.load(open("test_set.pkl", "rb"))
    prediction_columns = ["room_type", "accommodates", "bathrooms", "bedrooms", "beds"]
    metrics = {key: Metric() for key in prediction_columns}

    col_to_prefix = {
        "neighbourhood": "neighb:",
        "neighbourhood_cleansed": "neigh.cl:",
        "property_type": "prop_type:",
        "room_type": "room_type:",
        "accommodates": "accom:",
        "bathrooms": "bathrooms:",
        "bedrooms": "bedrooms:",
        "beds": "beds:",
    }
    skippable = ["neighbourhood", "neighbourhood_cleansed", "property_type"]
    for row in test_set.itertuples(index=False):
        context_list = []
        for col, item in zip(row._fields, row):
            if col in skippable:
                if pd.notna(item):
                    if col in col_to_prefix:
                        context_list.append(f"{col_to_prefix[col]}{item}")
                continue
            if pd.isna(item):
                metrics[col].nan += 1
            else:
                if isinstance(predictor, AprioriPredictor):
                    prefix = col_to_prefix.get(col, "")
                    predicted = predictor.predict(context_list, prefix)
                    target_item = f"{prefix}{item}"
                    if target_item in predicted:
                        metrics[col].correct += 1
                    else:
                        metrics[col].wrong += 1
                else:
                    predicted = predictor.predict(col)
                    if item in predicted:
                        metrics[col].correct += 1
                    else:
                        metrics[col].wrong += 1

            if pd.notna(item):
                if col in col_to_prefix:
                    context_list.append(f"{col_to_prefix[col]}{item}")

    for key in metrics.keys():
        print(key, metrics[key])


def main():
    clean_data()
    df = pickle.load(open("data.pkl", "rb"))
    split_data(df)
    hPredictor, aPredictor = train()
    print("--- Heuristic Results ---")
    validate(hPredictor)
    print("\n--- Apriori Results ---")
    validate(aPredictor)


if __name__ == "__main__":
    main()
