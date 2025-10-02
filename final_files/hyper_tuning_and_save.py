import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from preprocessing import load_and_preprocess
from final_files.base_model import build_base_pipe

# 1. Load data
X, y, features = load_and_preprocess("ErWaitTime.csv")

# Define target names
targets = [
    "Total Wait Time (min)",
    "Time to Registration (min)",
    "Time to Triage (min)",
    "Time to Medical Professional (min)"
]

# 2. Base pipeline with MultiOutputRegressor
base_regressor = build_base_pipe(features).named_steps['regressor']
pipe = Pipeline(steps=[
    ('preprocessor', build_base_pipe(features).named_steps['preprocessor']),
    ('regressor', MultiOutputRegressor(base_regressor))
])

# 3. Hyperparameter grid
param_grid = {
    'regressor__estimator__n_estimators': [200, 400],
    'regressor__estimator__max_depth': [3, 5, 7],
    'regressor__estimator__learning_rate': [0.05, 0.1, 0.2],
    'regressor__estimator__subsample': [0.7, 0.9, 1.0],
    'regressor__estimator__colsample_bytree': [0.7, 0.9, 1.0]
}

# 4. GridSearch
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)

search.fit(X, y)
print("BEST PARAMS:", search.best_params_)

# 5. Train/Test evaluate
best_pipe = search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

# Evaluate each target
for i, target in enumerate(targets):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\nMetrics for {target}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

# 6. Identify bottleneck (max time among Registration, Triage, Medical Professional)
bottleneck_steps = ["Registration", "Triage", "Medical Professional"]
y_pred_bottleneck = y_pred[:, 1:4]  # Exclude Total Wait Time for bottleneck analysis
bottleneck_indices = np.argmax(y_pred_bottleneck, axis=1)
bottlenecks = [bottleneck_steps[idx] for idx in bottleneck_indices]

# Create a DataFrame to show bottleneck for each test instance
results_df = pd.DataFrame({
    "Predicted Registration Time (min)": y_pred[:, 1],
    "Predicted Triage Time (min)": y_pred[:, 2],
    "Predicted Medical Professional Time (min)": y_pred[:, 3],
    "Bottleneck": bottlenecks
})

# Summarize bottleneck frequency
bottleneck_counts = results_df["Bottleneck"].value_counts()
print("\nBottleneck Frequency:")
print(bottleneck_counts)

# 7. Save tuned model
joblib.dump(best_pipe, "../hospitalAi/tuned_multioutput_xgboost.pkl")
print("Saved tuned model to tuned_multioutput_xgboost.pkl")

# Save bottleneck results
results_df.to_csv("./hospitalAi/bottleneck_results.csv", index=False)
print("Saved bottleneck analysis to bottleneck_results.csv")