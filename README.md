# 🍔 Zomato Delivery AI

A simple 2–3 hour classroom project demonstrating **Linear Regression + Logistic Regression** on the same food-delivery business problem.

## Business story

> **Can AI tell us both how long an order will take and whether it is likely to be late?**

### Model 1 — ETA AI
**Linear Regression** predicts delivery time in minutes.

Target:
`Time_taken (min)`

### Model 2 — Late Delivery AI
**Logistic Regression** predicts whether delivery is likely to be late.

Teaching rule:
`late = Time_taken (min) > 30`

## Shared inputs

- Distance (km)
- Delivery person age
- Delivery person rating
- Vehicle condition
- Additional deliveries
- Traffic density
- Weather
- Festival

The code also calculates `distance_km` from restaurant/customer GPS coordinates when that engineered column is not already present.

## Project files

```text
zomato_delivery_ai/
├── zomato_cleaned.csv          # Put the dataset here
├── train_models.py             # preprocessing + training + evaluation + .pkl export
├── app.py                      # Streamlit app
├── notebook_outline.md         # classroom notebook sequence
├── requirements.txt
└── models/
    ├── eta_linear_regression.pkl
    ├── late_logistic_regression.pkl
    └── model_info.pkl
```

## Run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Important teaching choice

Do **not** feed the actual delivery time into the features. It is the target. This keeps the demo free from target leakage.

The logistic model also does **not** use the linear model's prediction as an input. Both models independently learn from the same operational features, which makes the comparison cleaner for teaching.

## 2–3 hour class flow

**0:00–0:20** Business problem + dataset + simple EDA  
**0:20–0:45** Cleaning + distance feature + late label  
**0:45–1:20** Linear Regression + metrics  
**1:20–1:55** Logistic Regression + confusion matrix  
**1:55–2:15** Save/load `.pkl` files  
**2:15–3:00** Streamlit app + demo + small student exercises
