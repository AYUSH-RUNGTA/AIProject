# Food Delivery Time Predictor
# Predict how many minutes a food delivery takes.
# Compares two models: Random Forest vs Neural Network.
# Run:  python main.py

import os
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---- 1. Load data ----
path = kagglehub.dataset_download("gauravmalik26/food-delivery-dataset")
df = pd.read_csv(os.path.join(path, "train.csv"))
print("Rows loaded:", len(df))


# ---- 2. Clean data ----
# Keep only the columns we need, with short names
df = df.rename(columns={
    "Delivery_person_Age": "age",
    "Delivery_person_Ratings": "rating",
    "Vehicle_condition": "vehicle_condition",
    "multiple_deliveries": "multiple_deliveries",
    "Weatherconditions": "weather",
    "Road_traffic_density": "traffic",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle",
    "Festival": "festival",
    "City": "city",
    "Time_taken(min)": "time_taken",
})
df = df[["age", "rating", "vehicle_condition", "multiple_deliveries", "weather",
         "traffic", "order_type", "vehicle", "festival", "city", "time_taken"]]

# Text columns have extra spaces, e.g. "High " -> "High"
for col in df.columns:
    if df[col].dtype != "int64":
        df[col] = df[col].astype(str).str.strip()

# "(min) 24" -> 24   and   "conditions Sunny" -> "Sunny"
df["time_taken"] = df["time_taken"].str.replace("(min) ", "").astype(int)
df["weather"] = df["weather"].str.replace("conditions ", "")

# Missing values are written as the text "NaN" -> drop those rows
df = df.replace("NaN", None).dropna()

# Numbers stored as text -> real numbers
df["age"] = df["age"].astype(int)
df["rating"] = df["rating"].astype(float)
df["multiple_deliveries"] = df["multiple_deliveries"].astype(float).astype(int)
print("Rows after cleaning:", len(df))


# ---- 3. Explore data ----
print("\nAverage delivery time:", round(df["time_taken"].mean(), 1), "min")
print("\nAverage time by traffic:")
print(df.groupby("traffic")["time_taken"].mean().round(1))
print("\nAverage time by weather:")
print(df.groupby("weather")["time_taken"].mean().round(1))

# Plots are kept in variables (fig1, fig2, ...) so the website can show them
fig1, ax1 = plt.subplots()
df["time_taken"].plot(kind="hist", bins=30, ax=ax1, title="How long do deliveries take? (minutes)")

fig2, ax2 = plt.subplots()
df.groupby("traffic")["time_taken"].mean().plot(kind="bar", ax=ax2, title="Average time by traffic")

fig3, ax3 = plt.subplots()
df.groupby("weather")["time_taken"].mean().plot(kind="bar", ax=ax3, title="Average time by weather")


# ---- 4. Prepare data for the models ----
X = pd.get_dummies(df.drop(columns="time_taken"))  # text columns -> 0/1 columns
y = df["time_taken"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural networks need numbers on the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---- 5. Train both models ----
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
nn.fit(X_train, y_train)


# ---- 6. Test and compare ----
rf_error = mean_absolute_error(y_test, rf.predict(X_test))
nn_error = mean_absolute_error(y_test, nn.predict(X_test))
rf_r2 = r2_score(y_test, rf.predict(X_test))
nn_r2 = r2_score(y_test, nn.predict(X_test))

print("\n---- Results (lower error is better) ----")
print(f"Random Forest : off by {rf_error:.2f} min on average, R2 = {rf_r2:.3f}")
print(f"Neural Network: off by {nn_error:.2f} min on average, R2 = {nn_r2:.3f}")

if rf_error < nn_error:
    winner = "Random Forest"
else:
    winner = "Neural Network"
print("Winner:", winner)

results = {"Random Forest": round(rf_error, 2), "Neural Network": round(nn_error, 2)}

fig4, ax4 = plt.subplots()
pd.Series(results).plot(kind="bar", ax=ax4, color=["tab:green", "tab:orange"],
                        title="Average error in minutes (lower is better)")


# ---- 7. Show the plots (only when running main.py directly) ----
if __name__ == "__main__":
    plt.show()
