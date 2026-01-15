import pickle


def extract_samples():
    try:
        df = pickle.load(open("test_set.pkl", "rb"))

        complete_records = df.dropna()

        if len(complete_records) == 0:
            print(
                "Warning: No fully complete records found in test set. Showing samples from best available data."
            )
            samples = df.sample(5)
        else:
            n_samples = min(5, len(complete_records))
            samples = complete_records.sample(n_samples)

        print("# Manual Test Data Scenarios (Complete Rows)\n")
        print(
            "These samples have data for ALL fields, making them perfect for verifying predictions.\n"
        )

        for index, row in samples.iterrows():
            print(f"## Scenario {index}")
            print("| Field | Value |")
            print("|---|---|")
            for col in df.columns:
                val = row[col]
                print(f"| **{col}** | {val} |")
            print("\n")

    except FileNotFoundError:
        print(
            "Error: test_set.pkl not found. Make sure you have run the cleaning script first."
        )
    except Exception as e:
        print(f"Error extracting samples: {e}")


if __name__ == "__main__":
    extract_samples()
