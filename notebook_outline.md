# Zomato Delivery AI — 2–3 Hour Classroom Notebook

## Business problem
For every food order, the operations team wants two answers:

1. **ETA AI (Regression):** How many minutes will this order take?
2. **Late Delivery AI (Classification):** Is this order likely to be late?

The two models use the same order/delivery features.

## 1. Load data
```python
import pandas as pd

df = pd.read_csv('zomato_cleaned.csv')
df.head()
df.shape
```

## 2. Minimal EDA
```python
df.info()
df.isna().sum().sort_values(ascending=False).head(10)
df['Time_taken_min'].describe()
```

One histogram:
```python
import matplotlib.pyplot as plt

df['Time_taken_min'].hist(bins=20)
plt.xlabel('Delivery time (min)')
plt.ylabel('Orders')
plt.show()
```

One business comparison:
```python
df.groupby('Road_traffic_density')['Time_taken_min'].mean().sort_values()
```

## 3. Targets
The clean dataset contains:
- `Time_taken_min`: Continuous regression target
- `late`: Binary classification target (`1` if `Time_taken_min > 30`, else `0`)

Explain: **30 minutes is the business cutoff for this teaching demo.**

## 4. Features
Use only the 8 operational inputs:
```python
features = [
    'distance_km',
    'Delivery_person_Age',
    'Delivery_person_Ratings',
    'Vehicle_condition',
    'multiple_deliveries',
    'Road_traffic_density',
    'Weather_conditions',
    'Festival'
]
```

## 5. Train/test split
Use one split for both models so the comparison is fair.

```python
from sklearn.model_selection import train_test_split
X = df[features]
y_eta = df['Time_taken_min']
y_late = df['late']

X_train, X_test, y_eta_train, y_eta_test, y_late_train, y_late_test = train_test_split(
    X, y_eta, y_late, test_size=0.2, random_state=42, stratify=y_late
)
```

## 6. Preprocessing: Convert Text to Numbers (`pd.get_dummies`)
Models cannot do math with words. We use `pd.get_dummies()` to create 0/1 columns:

```python
# One-hot encode text categories
df = pd.get_dummies(df, columns=['Road_traffic_density', 'Weather_conditions', 'Festival'], dtype=int)

# Separate inputs (X) and targets (y)
X = df.drop(columns=['Time_taken_min', 'late'])
y_eta = df['Time_taken_min']
y_late = df['late']

# 80/20 train/test split
X_train, X_test, y_eta_train, y_eta_test, y_late_train, y_late_test = train_test_split(
    X, y_eta, y_late, test_size=0.2, random_state=42
)
```

## 7. Linear Regression (ETA AI)
```python
from sklearn.linear_model import LinearRegression

eta_model = LinearRegression()
eta_model.fit(X_train, y_eta_train)
pred_eta = eta_model.predict(X_test)
```

Metric:
- MAE (Mean Absolute Error) — average minutes off

## 8. Logistic Regression (Late Delivery AI)
```python
from sklearn.linear_model import LogisticRegression

late_model = LogisticRegression(max_iter=1000)
late_model.fit(X_train, y_late_train)
pred_late = late_model.predict(X_test)
```

Metrics:
- Accuracy
- Precision
- Recall
- F1

Confusion matrix:
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_late_test, pred_late)
```

## 9. Save models
```python
import os
import pickle

os.makedirs('models', exist_ok=True)

with open('models/eta_linear_regression.pkl', 'wb') as f:
    pickle.dump(eta_model, f)

with open('models/late_logistic_regression.pkl', 'wb') as f:
    pickle.dump(late_model, f)

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump({'features': features, 'late_threshold': 30}, f)
```

## 10. Streamlit deployment
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The app asks for the same eight inputs and displays both predictions side-by-side.
