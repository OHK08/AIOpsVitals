import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Parse Visit Date to ensure it's datetime
    df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")
    df = df.dropna(subset=["Visit Date"])

    # Define target columns
    targets = [
        "Total Wait Time (min)",
        "Time to Registration (min)",
        "Time to Triage (min)",
        "Time to Medical Professional (min)"
    ]

    # Check if all target columns exist
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns in dataset: {missing_targets}")

    # Select feature columns (excluding Visit ID, Patient ID, Hospital ID, Hospital Name)
    features = [
        "Region",
        "Day of Week",
        "Season",
        "Time of Day",
        "Urgency Level",
        "Nurse-to-Patient Ratio",
        "Specialist Availability",
        "Facility Size (Beds)"
    ]

    # Check if all feature columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns in dataset: {missing_features}")

    X = df[features]
    y = df[targets]
    return X, y, features