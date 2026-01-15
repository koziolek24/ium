from fastapi import FastAPI, Response, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import random
from clean import train, AprioriPredictor


class FormData(BaseModel):
    neighbourhood: Optional[str] = None
    neighbourhood_cleansed: Optional[str] = None
    property_type: Optional[str] = None
    accommodates: Optional[str] = None
    bathrooms: Optional[str] = None
    beds: Optional[str] = None
    bedrooms: Optional[str] = None
    room_type: Optional[str] = None


class MetricData(BaseModel):
    duration_seconds: float

app = FastAPI()

apriori_model = None
heuristic_model = None

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


@app.on_event("startup")
def load_models():
    global apriori_model, heuristic_model
    hPredictor, aPredictor = train()
    apriori_model = aPredictor
    heuristic_model = hPredictor
    print("Models loaded: Apriori and Heuristic ready.")


@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.post("/submit_time")
def submit_time(metric: MetricData, request: Request):
    model_version = request.cookies.get("model_version", "Unknown")

    log_entry = f"Model: {model_version}, Duration: {metric.duration_seconds}s"
    print(f"METRIC: {log_entry}")

    with open("ab_results.txt", "a") as f:
        f.write(f"{model_version},{metric.duration_seconds}\n")

    return {"status": "recorded"}


@app.post("/predict")
def predict(form_data: FormData, response: Response, request: Request):
    if not apriori_model or not heuristic_model:
        return {"error": "Models not loaded"}

    model_choice = request.cookies.get("model_version")

    if not model_choice:
        model_choice = random.choice(["A", "B"])
        response.set_cookie(key="model_version", value=model_choice)

    if model_choice == "A":
        active_predictor = apriori_model
        predictor_name = "Apriori"
    else:
        active_predictor = heuristic_model
        predictor_name = "Heuristic"

    print(f"Request handled by: {predictor_name} (Group {model_choice})")

    context_list = []
    data_dict = form_data.model_dump(exclude_none=True)

    for col, value in data_dict.items():
        if value and col in col_to_prefix:
            prefix = col_to_prefix[col]
            if not value.startswith(prefix):
                context_list.append(f"{prefix}{value}")
            else:
                context_list.append(value)

    predictions = {}

    for field_name in form_data.model_fields.keys():
        if getattr(form_data, field_name):
            continue

        if field_name in col_to_prefix:
            results = None
            prefix = col_to_prefix[field_name]

            if isinstance(active_predictor, AprioriPredictor):
                results = active_predictor.predict(context_list, prefix)
            else:
                results = active_predictor.predict(field_name)

            if results:
                best_match = list(results)[0]

                if best_match.startswith(prefix):
                    clean_value = best_match[len(prefix) :]
                    predictions[field_name] = clean_value
                else:
                    predictions[field_name] = best_match

    predictions["_model_used"] = predictor_name

    return predictions


if __name__ == "__main__":
    import uvicorn

    load_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)
