from collections import defaultdict


class HeuristicPredictor:
    def __init__(self, df):
        self.predictionSet = defaultdict(list)
        self.fit(df)

    def fit(self, df):
        for col in df.columns:
            self.predictionSet[col] = (
                df[col].value_counts(dropna=True).head(1).index.to_list()
            )

    def predict(self, key):
        return self.predictionSet.get(key, None)
