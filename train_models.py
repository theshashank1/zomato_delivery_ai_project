import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

DATA_PATH = "zomato_cleaned.csv"
OUT_DIR = "models"
LATE_THRESHOLD = 30


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load clean dataset (no missing values)
    df = pd.read_csv(DATA_PATH).dropna()

    # 2. Convert text columns to numbers (One-Hot Encoding with pd.get_dummies)
    df = pd.get_dummies(df, columns=["Road_traffic_density", "Weather_conditions", "Festival"], dtype=int)

    # 3. Separate features (X) and targets (y)
    X = df.drop(columns=["Time_taken_min", "late"])
    y_eta = df["Time_taken_min"]
    y_late = df["late"]

    feature_cols = list(X.columns)

    # 4. Split data (80% training, 20% testing)
    X_train, X_test, y_eta_train, y_eta_test, y_late_train, y_late_test = train_test_split(
        X, y_eta, y_late, test_size=0.2, random_state=42
    )

    # 5. Model 1 — Linear Regression (Predict Delivery Time)
    eta_model = LinearRegression()
    eta_model.fit(X_train, y_eta_train)
    eta_pred = eta_model.predict(X_test)

    # 6. Model 2 — Logistic Regression (Predict Late Risk)
    late_model = LogisticRegression(max_iter=1000)
    late_model.fit(X_train, y_late_train)
    late_pred = late_model.predict(X_test)

    # 7. Print Performance Metrics
    print("=== ETA AI (Linear Regression) ===")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_eta_test, eta_pred):.2f} minutes")

    print(f"\n=== Late Delivery AI (Logistic Regression, > {LATE_THRESHOLD} min) ===")
    print(f"Accuracy: {accuracy_score(y_late_test, late_pred) * 100:.1f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_late_test, late_pred))

    # 8. Save models and feature columns list
    with open(os.path.join(OUT_DIR, "eta_linear_regression.pkl"), "wb") as f:
        pickle.dump(eta_model, f)
    with open(os.path.join(OUT_DIR, "late_logistic_regression.pkl"), "wb") as f:
        pickle.dump(late_model, f)
    with open(os.path.join(OUT_DIR, "model_info.pkl"), "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "late_threshold": LATE_THRESHOLD}, f)

    print("\nSaved:")
    print("  models/eta_linear_regression.pkl")
    print("  models/late_logistic_regression.pkl")
    print("  models/model_info.pkl")


if __name__ == "__main__":
    main()
