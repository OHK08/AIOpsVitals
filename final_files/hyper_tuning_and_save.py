import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from preprocessing import load_and_preprocess
from final_files.base_model import build_base_pipe


# -------------------------------------------------
# 1. Load and preprocess data
# -------------------------------------------------
X, y, features = load_and_preprocess("ErWaitTime.csv")

targets = [
    "Total Wait Time (min)",
    "Time to Registration (min)",
    "Time to Triage (min)",
    "Time to Medical Professional (min)"
]


# -------------------------------------------------
# 2. Build pipeline
# -------------------------------------------------
base_pipe = build_base_pipe(features)

pipe = Pipeline(steps=[
    ("preprocessor", base_pipe.named_steps["preprocessor"]),
    ("regressor", MultiOutputRegressor(base_pipe.named_steps["regressor"]))
])


# -------------------------------------------------
# 3. Hyperparameter grid
# -------------------------------------------------
param_grid = {
    "regressor__estimator__n_estimators": [200, 400],
    "regressor__estimator__max_depth": [3, 5, 7],
    "regressor__estimator__learning_rate": [0.05, 0.1, 0.2],
    "regressor__estimator__subsample": [0.7, 0.9, 1.0],
    "regressor__estimator__colsample_bytree": [0.7, 0.9, 1.0]
}


# -------------------------------------------------
# 4. Grid search
# -------------------------------------------------
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)

search.fit(X, y)
best_pipe = search.best_estimator_

print("BEST PARAMETERS:")
print(search.best_params_)


# -------------------------------------------------
# 5. Train-test split and prediction
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)


# -------------------------------------------------
# 6. Regression performance metrics
# -------------------------------------------------
print("\n===== REGRESSION PERFORMANCE =====")

for i, target in enumerate(targets):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

    print(f"\n{target}")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")


# -------------------------------------------------
# 7. Data-driven threshold accuracy
# -------------------------------------------------
print("\n===== DATA-DRIVEN THRESHOLD ACCURACY =====")

for i, target in enumerate(targets):
    abs_error = np.abs(y_test.iloc[:, i].values - y_pred[:, i])

    median_thresh = np.percentile(abs_error, 50)
    upper_thresh = np.percentile(abs_error, 75)

    acc_median = np.mean(abs_error <= median_thresh)
    acc_upper = np.mean(abs_error <= upper_thresh)

    print(f"\n{target}")
    print(f"Accuracy within median error (±{median_thresh:.2f} min): {acc_median*100:.2f}%")
    print(f"Accuracy within 75th percentile error (±{upper_thresh:.2f} min): {acc_upper*100:.2f}%")


# -------------------------------------------------
# 8. Data-driven categorization for confusion matrix
# -------------------------------------------------
def categorize_wait_time(values):
    q1 = np.percentile(values, 33)
    q2 = np.percentile(values, 66)

    return pd.cut(
        values,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"]
    )


# -------------------------------------------------
# 9. Confusion matrix – publication-quality plots
# -------------------------------------------------
print("\n===== CONFUSION MATRIX ANALYSIS =====")

class_labels = ["Low", "Medium", "High"]

for i, target in enumerate(targets):
    y_true_cat = categorize_wait_time(y_test.iloc[:, i])
    y_pred_cat = categorize_wait_time(y_pred[:, i])

    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=class_labels)

    print(f"\nConfusion Matrix – {target}")
    print(cm)
    print(classification_report(y_true_cat, y_pred_cat))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    im = ax.imshow(cm)

    ax.set_title(f"{target}", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("Actual Class", fontsize=10)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)

    # Annotate values
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, cm[r, c],
                    ha="center", va="center",
                    fontsize=9)

    fig.tight_layout()
    plt.savefig(f"confusion_matrix_{target.replace(' ', '_')}.png")
    plt.show()


# -------------------------------------------------
# 10. Bottleneck identification
# -------------------------------------------------
bottleneck_steps = ["Registration", "Triage", "Medical Professional"]

y_pred_bottleneck = y_pred[:, 1:4]
bottleneck_indices = np.argmax(y_pred_bottleneck, axis=1)
bottlenecks = [bottleneck_steps[idx] for idx in bottleneck_indices]

results_df = pd.DataFrame({
    "Predicted Registration Time (min)": y_pred[:, 1],
    "Predicted Triage Time (min)": y_pred[:, 2],
    "Predicted Medical Professional Time (min)": y_pred[:, 3],
    "Bottleneck": bottlenecks
})

print("\nBottleneck Frequency:")
print(results_df["Bottleneck"].value_counts())


# -------------------------------------------------
# 11. Save model and results
# -------------------------------------------------
joblib.dump(best_pipe, "../hospitalAi/tuned_multioutput_xgboost.pkl")
results_df.to_csv("./hospitalAi/bottleneck_results.csv", index=False)

print("\nModel and outputs saved successfully.")
