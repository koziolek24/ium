# IUM Maciej Kozlowski Marek Dzieciol

## Prerequisites

*   Python 3.13+
*   uv (package manager)

## Setup and Installation

This project uses `uv` for dependency management.

1.  **Sync dependencies**:
    ```bash
    uv sync
    ```

    This creates the virtual environment and installs all required packages as defined in `pyproject.toml`.

## Running the Application

1.  **Start the server**:
    ```bash
    uv run fastapi dev main.py
    ```

2.  **Access the interface**:
    Open `http://127.0.0.1:8000` in your web browser.

## Features

*   **Prediction Models**:
    *   **Apriori**: Context-aware association rules.
    *   **Heuristic**: Column-based logic (baseline).
*   **A/B Testing**:
    *   Users are randomly assigned to Group A (Apriori) or Group B (Heuristic).
    *   Assignments are sticky per browser session via cookies.
*   **Metrics**:
    *   Time-To-Completion is tracked from the first input interaction until submission.
    *   Results are logged to `ab_results.txt`.

## Data Preparation

To regenerate the `data.pkl`, `test_set.pkl`, and `train_set.pkl` artifacts:

```bash
uv run python clean.py
```
## A/B Tests

To generate a/b tests data as ab_results.txt:
```bash
python ab_tests_generator.py
```

To analyze a/b tests results:
```bash
python ab_tests_analyzer.py
```

To analyze our a/b tests results:
```bash
python ab_tests_analyzer.py ab_results.csv
```
