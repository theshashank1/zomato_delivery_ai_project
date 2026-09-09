# 🍔 Zomato Delivery AI — Classroom Project

A beginner-friendly 2–3 hour machine learning project for 1st year B.Tech students demonstrating **Linear Regression** (Regression) and **Logistic Regression** (Classification) on the same food-delivery business problem.

---

## 🎯 Business Story

> **"When a customer places an order on Zomato, can AI tell us both how long it will take and whether it is likely to be delayed?"**

### 1. Model 1 — ETA AI (Linear Regression)
* **Goal:** Predict estimated delivery time in minutes.
* **Target:** `Time_taken_min` (Continuous number, e.g., 24.5 min).

### 2. Model 2 — Late Delivery AI (Logistic Regression)
* **Goal:** Predict whether the delivery is likely to exceed the 30-minute threshold.
* **Target:** `late` (Binary: `1` if `Time_taken_min > 30`, else `0`).

---

## 📋 The 10 Clean Columns in `zomato_cleaned.csv`

1. `distance_km` — Distance between restaurant and customer (km)
2. `Delivery_person_Age` — Age of the delivery partner
3. `Delivery_person_Ratings` — Average rating of the rider (1.0 to 5.0)
4. `Vehicle_condition` — Condition of the vehicle (0 = poor, 1 = fair, 2 = good)
5. `multiple_deliveries` — Number of additional orders assigned to the rider (0 to 3)
6. `Road_traffic_density` — Traffic level (`Low`, `Medium`, `High`, `Jam`)
7. `Weather_conditions` — Weather (`Sunny`, `Cloudy`, `Fog`, `Stormy`, `Windy`, `Sandstorms`)
8. `Festival` — Festival season (`Yes` or `No`)
9. `Time_taken_min` — Actual delivery time in minutes (Regression Target)
10. `late` — Late indicator (Classification Target)

---

## 📁 Project Structure

```text
zomato_delivery_ai_project/
├── zomato_cleaned.csv          # Clean dataset (42,444 orders, 10 columns, 0 nulls)
├── zomato_delivery_ai.ipynb    # Main demonstration Jupyter notebook for class
├── notebook_outline.md         # Teacher's lecture guide & talking points
├── train_models.py             # Standalone Python script to train & export models
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Python dependencies
└── models/
    ├── eta_linear_regression.pkl   # Trained Linear Regression model
    ├── late_logistic_regression.pkl # Trained Logistic Regression model
    └── model_info.pkl              # Feature columns & metadata
```

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (or run the notebook)
python train_models.py

# 3. Launch the interactive web app
streamlit run app.py
```

---

## ⏰ 2–3 Hour Class Schedule

* **0:00 – 0:20 | Introduction & Business Problem**
  * Why food delivery apps need AI.
  * Regression (How many minutes?) vs. Classification (Is it late? Yes/No).
* **0:20 – 0:45 | Data Exploration & Preprocessing**
  * Loading `zomato_cleaned.csv`.
  * Visualizing delivery times with a 2-line histogram.
  * Converting text categories to numbers using `pd.get_dummies()`.
* **0:45 – 1:20 | Model 1: Linear Regression (ETA AI)**
  * Fitting the model in 3 lines of code.
  * Understanding Mean Absolute Error (MAE ~ 4.8 minutes).
* **1:20 – 1:55 | Model 2: Logistic Regression (Late Delivery AI)**
  * Predicting binary late risk.
  * Understanding Accuracy (~88%) and the Confusion Matrix.
* **1:55 – 2:15 | Testing on a Live Order & Saving Models**
  * Simulating a live customer order with custom inputs.
  * Saving models with `pickle.dump()`.
* **2:15 – 3:00 | Live Streamlit Demo & Q&A**
  * Running `streamlit run app.py`.
  * Letting students test different traffic/weather combinations.
