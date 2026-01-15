from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import generate_rules_apriori
import pandas as pd


class AprioriPredictor:
    def __init__(self, df, min_confidence, min_support):
        transactions = self.parse_data(df)
        self.itemsets, self.num_transactions = itemsets_from_transactions(
            transactions, min_support, output_transaction_ids=True
        )
        itemsets_raw = {
            length: {
                item: counter.itemset_count for (item, counter) in itemsets.items()
            }
            for (length, itemsets) in self.itemsets.items()
        }
        self.rules = list(
            generate_rules_apriori(itemsets_raw, min_confidence, self.num_transactions)
        )

    def predict(self, context, target_prefix):
        predictions = set()
        context_set = set(context)

        for rule in self.rules:
            if set(rule.lhs).issubset(context_set):
                for item in rule.rhs:
                    if item.startswith(target_prefix):
                        predictions.add(item)

        return list(predictions)

    def clean_row(self, row):
        clean_row = []
        if pd.notna(row.neighbourhood):
            clean_row.append(f"neighb:{row.neighbourhood}")
        if pd.notna(row.neighbourhood_cleansed):
            clean_row.append(f"neigh.cl:{row.neighbourhood_cleansed}")
        if pd.notna(row.property_type):
            clean_row.append(f"prop_type:{row.property_type}")
        if pd.notna(row.room_type):
            clean_row.append(f"room_type:{row.room_type}")
        if pd.notna(row.accommodates):
            clean_row.append(f"accom:{row.accommodates}")
        if pd.notna(row.bathrooms):
            clean_row.append(f"bathrooms:{row.bathrooms}")
        if pd.notna(row.bedrooms):
            clean_row.append(f"bedrooms:{row.bedrooms}")
        if pd.notna(row.beds):
            clean_row.append(f"beds:{row.beds}")
        return tuple(clean_row)

    def parse_data(self, df):
        transactions = []
        for row in df.itertuples(index=False):
            transactions.append(self.clean_row(row))

        transactions = tuple(transactions)
        return transactions
