import pandas as pd
import numpy as np
import sys

FILE = sys.argv[1] if len(sys.argv) > 1 else "ab_results.txt"
TOTAL_FIELDS = 8

df = pd.read_csv(
    FILE,
    header=None,
    names = ['group', 'duration_seconds', 'fields_predicted_before_fill'],
)

ab_filter = df['group'].isin(['A', 'B'])
other = df[~ab_filter]
df = df[ab_filter].copy()

df['fields_user_typed'] = TOTAL_FIELDS - df['fields_predicted_before_fill']
df['fields_user_typed'] = df['fields_user_typed'].clip(lower=0, upper=TOTAL_FIELDS)

def summarize(data: pd.DataFrame) -> pd.Series:
    d = data["duration_seconds"].to_numpy()
    t = data["fields_user_typed"].to_numpy()
    p = data["fields_predicted_before_fill"].to_numpy()
    return pd.Series({
        "n": len(data),
        "duration_mean": d.mean(),
        "duration_median": np.median(d),
        "duration_p90": np.quantile(d, 0.90),
        "typed_mean": t.mean(),
        "typed_median": np.median(t),
        "autofill_mean": p.mean(),
        "autofill_median": np.median(p),
    })

print('A = Apriori; B = Heurystyka')

summary = df.groupby('group', sort=True).apply(summarize, include_groups=False).round(2)
print(summary.to_string())

A, B = df[df.group == "A"], df[df.group == "B"]
diff_duration_mean = A["duration_seconds"].mean() - B["duration_seconds"].mean()
diff_typed_mean = A["fields_user_typed"].mean() - B["fields_user_typed"].mean()
print("\nA - B:")
print(f"  duration_mean_difference: {diff_duration_mean:.3f} s")
print(f"  typed_mean_difference:    {diff_typed_mean:.3f} fields")

if len(other) > 0:
    print(f"\nINFO: pominięto {len(other)} wierszy z group != A/B")