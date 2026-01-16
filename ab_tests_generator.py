import time, pickle, requests
import pandas as pd

BASE = "http://127.0.0.1:8000"
PREDICT = f"{BASE}/predict"
SUBMIT = f"{BASE}/submit_time"


FIELDS = [
    "neighbourhood",
    "neighbourhood_cleansed",
    "property_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "room_type",
]

open("ab_results.txt", "w").close()
print("ab_results.txt cleared")

client_data = pickle.load(open("test_set.pkl", "rb")) 

iters = 0
total = len(client_data)
for _, row in client_data.iterrows():
    s = requests.Session()
    form = {k: "" for k in FIELDS}
    predicted_before_fill = set()
    start = None

    for f in FIELDS:
        if start is None:
            start = time.time()
        
        if pd.notna(row[f]) and form[f] != "" and form[f] != str(row[f]):
            predicted_before_fill.discard(f)
        
        if pd.notna(row[f]):
            form[f] = str(row[f])
        
        payload = {k: ("" if form[k] == "" else str(form[k])) for k in FIELDS}
        r = s.post(PREDICT, json=payload, timeout=30)
        if r.status_code in [422, 500]:
            raise RuntimeError(r.text)
        r.raise_for_status()
        preds = r.json()

        for k, v in preds.items():
            if k.startswith('_') or not v:
                continue
            if form.get(k, "") == "":
                predicted_before_fill.add(k)
                form[k] = v
    
    # dodatkowy czas = 
    # + za każdy wpisany znak (0.5s) 
    # + 1 sekunda za każde nie-uzupełnione pole (użytkownik musi przełączyć kursor i się zastanowić co wpisać)
    total_chars = sum(len(str(form[k])) for k in FIELDS)
    predicted_chars = sum(len(str(form[k])) for k in predicted_before_fill)
    typed_chars = total_chars - predicted_chars
    user_filed_fills = len(FIELDS) - len(predicted_before_fill)
    extra_time = 0.5 * typed_chars + 1 * user_filed_fills

    s.post(
        SUBMIT,
        json = {
            "duration_seconds": float(time.time() - start) + extra_time,
            "fields_predicted_before_fill": len(predicted_before_fill),
        },
        timeout=30,
    ).raise_for_status()

    iters += 1
    print(f"\rProgres: {100*iters/total:.1f}%", end="", flush=True)