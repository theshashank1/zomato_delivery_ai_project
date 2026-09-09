# 🍔 Zomato Delivery AI — Teacher's Classroom Lecture Guide
### *2–3 Hour Introductory Machine Learning Lecture for 1st Year B.Tech Students*

---

## 🎯 Lecture Objectives
By the end of this class, students will:
1. Understand the difference between **Regression** (predicting a quantity) and **Classification** (predicting a category/flag).
2. Know why machine learning requires numbers instead of text, and how `pd.get_dummies()` converts words to 0/1.
3. Understand the **Train / Test Split** (the 80/20 rule) to avoid testing on the same questions the model studied.
4. Train two models in Scikit-Learn: **Linear Regression** and **Logistic Regression**.
5. Connect trained models to an interactive **Streamlit** dashboard.

---

## ⏰ Suggested Class Timeline

| Time | Duration | Topic | Key Student Activity |
| :---: | :---: | :--- | :--- |
| **0:00 – 0:20** | 20 min | **The Business Problem** | Discussing Zomato app features |
| **0:20 – 0:45** | 25 min | **Data Loading & Simple EDA** | `df.head()`, `df.describe()`, Histogram |
| **0:45 – 1:05** | 20 min | **Converting Text to Numbers** | `pd.get_dummies()` & Train/Test Split |
| **1:05 – 1:40** | 35 min | **Model 1: Linear Regression** | Fitting model & interpreting MAE |
| **1:40 – 2:15** | 35 min | **Model 2: Logistic Regression** | Fitting classifier & Confusion Matrix |
| **2:15 – 2:35** | 20 min | **Live Order Test & Model Saving** | Testing custom inputs & `pickle.dump()` |
| **2:35 – 3:00** | 25 min | **Streamlit App Demo & Hands-on** | Running `streamlit run app.py` |

---

## 📖 Section-by-Section Teaching Guide

### Section 1: The Business Problem (0:00 – 0:20)
* **What to say:** "Whenever you order biryani on Zomato, two things appear on your screen:
  1. *'Estimated delivery: 28 mins'* (A specific number)
  2. If the restaurant is far and it's raining, Zomato warns: *'Deliveries are delayed due to weather'* (A late risk warning)."
* 💡 **Analogy for Students:**
  * **Regression** is like predicting a student's exact exam score (e.g., `87.5 marks`).
  * **Classification** is like predicting whether the student will **Pass or Fail** (e.g., `Pass`).
* **Ask the Class:** *"Can we use the delivery time to predict if an order will be late?"*  
  $\rightarrow$ *Answer:* No! That is **Target Leakage**. You only know the delivery time *after* the food has already arrived!

---

### Section 2: Loading Data & Minimal EDA (0:20 – 0:45)
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("zomato_cleaned.csv")
df.head()
```

* **Inspect summary statistics:**
```python
df.describe().round(2)
```

* **Check missing values:**
```python
df.isna().sum()
```
* 💡 **Teacher's Point:** "Our dataset has 42,444 orders and 0 missing values. Clean data saves hours of debugging!"

* **Minimal 2-Line Plot (Delivery Time Histogram):**
```python
plt.hist(df["Time_taken_min"], bins=20, color="#E23744", edgecolor="black")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Number of Orders")
plt.show()
```

---

### Section 3: Converting Text to Numbers (0:45 – 1:05)
* **What to say:** "Computers cannot compute math with words. If traffic is `'Jam'`, what is `'Jam' * 2.5`? It makes no sense! We must convert words into numbers."
* 💡 **Analogy for Students (The Light Switch):**
  * Think of One-Hot Encoding like a row of light switches.
  * If traffic is `Jam`, turn switch `Road_traffic_density_Jam` to `1` (ON).
  * Turn all other traffic switches (`Low`, `Medium`, `High`) to `0` (OFF).

```python
# Convert text columns into 0 and 1 columns
df = pd.get_dummies(df, columns=["Road_traffic_density", "Weather_conditions", "Festival"], dtype=int)
df.head()
```

* **Separating Features ($X$) from Targets ($y$):**
```python
X = df.drop(columns=["Time_taken_min", "late"])
y_eta = df["Time_taken_min"]
y_late = df["late"]
```

* **The 80/20 Train / Test Split:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_eta_train, y_eta_test, y_late_train, y_late_test = train_test_split(
    X, y_eta, y_late, test_size=0.2, random_state=42
)
```
* 💡 **Analogy for Students (Exam Preparation):**
  * `X_train` is the textbook / homework problems students study from.
  * `X_test` is the final exam with **new, unseen questions**.
  * If you test a model on the same data it memorized during training, you will never know if it actually learned!

---

### Section 4: Model 1 — Linear Regression (1:05 – 1:40)
* **What to say:** "Linear Regression draws the best-fitting straight line through data points: $y = mx + c$."

```python
from sklearn.linear_model import LinearRegression

eta_model = LinearRegression()
eta_model.fit(X_train, y_eta_train)
eta_predictions = eta_model.predict(X_test)
```

* **Evaluating with MAE (Mean Absolute Error):**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_eta_test, eta_predictions)
print(f"Average Error (MAE): {mae:.2f} minutes")
```
* 💡 **Teacher's Point:** "Explain what MAE means in plain English: *On average, our AI is only off by about 4.8 minutes.* For an order taking 25 minutes, being off by ~4 minutes is very practical for a customer!"

* **Minimal 2-Line Scatter Plot:**
```python
plt.scatter(y_eta_test[:200], eta_predictions[:200], alpha=0.5, color="#E23744")
plt.xlabel("Actual Minutes")
plt.ylabel("Predicted Minutes")
plt.show()
```

---

### Section 5: Model 2 — Logistic Regression (1:40 – 2:15)
* **What to say:** "Linear Regression predicts a continuous number. But what if we only want a probability of being late? That's where **Logistic Regression** comes in. It produces an S-curve (sigmoid) between 0% and 100%."

```python
from sklearn.linear_model import LogisticRegression

late_model = LogisticRegression(max_iter=1000)
late_model.fit(X_train, y_late_train)
late_predictions = late_model.predict(X_test)
```

* **Evaluating Accuracy & Confusion Matrix:**
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

acc = accuracy_score(y_late_test, late_predictions)
print(f"Accuracy: {acc * 100:.1f}%")

cm = confusion_matrix(y_late_test, late_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On-Time", "Late"])
disp.plot(cmap="Blues")
plt.show()
```

* 💡 **Classroom Discussion Question:**
  * *"Is an 88% accuracy good?"*
  * *"What is worse in food delivery? Warning a customer that their food will be late when it actually arrives on time? Or promising it will be on-time, but delivering 30 minutes late?"*

---

### Section 6: Live Order Simulation & Saving Models (2:15 – 2:35)
* **What to say:** "Let's simulate a real customer placing an order right now."

```python
sample_order = pd.DataFrame([{
    "distance_km": 7.0,
    "Delivery_person_Age": 28,
    "Delivery_person_Ratings": 4.6,
    "Vehicle_condition": 1,
    "multiple_deliveries": 0,
    "Road_traffic_density": "High",
    "Weather_conditions": "Rainy",
    "Festival": "No"
}])

# Align columns with training data
sample_encoded = pd.get_dummies(sample_order).reindex(columns=X.columns, fill_value=0)

print("Predicted ETA :", eta_model.predict(sample_encoded)[0], "minutes")
print("Late Risk %   :", late_model.predict_proba(sample_encoded)[0, 1] * 100, "%")
```

* **Save Models to Disk:**
```python
import os
import pickle

os.makedirs("models", exist_ok=True)

with open("models/eta_linear_regression.pkl", "wb") as f:
    pickle.dump(eta_model, f)

with open("models/late_logistic_regression.pkl", "wb") as f:
    pickle.dump(late_model, f)

with open("models/model_info.pkl", "wb") as f:
    pickle.dump({"feature_cols": list(X.columns), "late_threshold": 30}, f)
```

---

### Section 7: Launching the Streamlit Web Application (2:35 – 3:00)
* Open the terminal in front of the class and run:
```bash
streamlit run app.py
```
* **Teacher's Demo Tips:**
  1. Show how dragging the **Distance slider** increases the predicted ETA.
  2. Switch traffic to **Jam** and weather to **Stormy** — watch the **Late Probability** jump past 80%!
  3. Invite a student to enter their own mock order details.
