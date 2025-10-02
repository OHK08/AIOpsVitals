from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def build_base_pipe(feature_columns):
    categorical = [
        "Region",
        "Day of Week",
        "Season",
        "Time of Day",
        "Urgency Level"
    ]

    numeric = [
        "Nurse-to-Patient Ratio",
        "Specialist Availability",
        "Facility Size (Beds)"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )

    regressor = XGBRegressor(
        n_estimators=200,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    return pipe