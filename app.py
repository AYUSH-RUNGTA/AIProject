# Website for the Food Delivery Time Predictor
# Run:  streamlit run app.py

import sys
import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True  # don't create a __pycache__ folder when importing main.py

# Importing main.py runs the whole pipeline once (load, clean, explore, train, compare).
# After that we can use its variables: main.rf, main.nn, main.fig1, ...
with st.spinner("Training the models, please wait..."):
    import main

st.title("🛵 Food Delivery Time Predictor")

# ---- Model comparison ----
st.subheader("Which model is better?")
st.table(pd.DataFrame(main.results.items(), columns=["Model", "Error (min)"]))
st.success(f"Winner: {main.winner}")

# ---- Plots: pick one from the dropdown ----
st.subheader("Plots")
plots = {
    "Delivery time": main.fig1,
    "Time by traffic": main.fig2,
    "Time by weather": main.fig3,
    "Model errors": main.fig4,
}
choice = st.selectbox("Choose a plot", list(plots.keys()))
st.pyplot(plots[choice])

# ---- Prediction form ----
st.subheader("Predict a delivery")
age = st.slider("Rider age", 18, 40, 28)
rating = st.slider("Rider rating", 1.0, 5.0, 4.5)
vehicle_condition = st.slider("Vehicle condition (0 = poor, 3 = great)", 0, 3, 1)
multiple_deliveries = st.slider("Other deliveries on the same trip", 0, 3, 1)
weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
festival = st.selectbox("Festival day?", ["No", "Yes"])
city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"])

if st.button("Predict"):
    order = pd.DataFrame([{
        "age": age, "rating": rating, "vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries, "weather": weather, "traffic": traffic,
        "order_type": order_type, "vehicle": vehicle, "festival": festival, "city": city,
    }])
    # Same steps as in main.py: text -> 0/1 columns, then scale
    order = pd.get_dummies(order).reindex(columns=main.X.columns, fill_value=0)
    order = main.scaler.transform(order)

    st.write(f"Random Forest: **{main.rf.predict(order)[0]:.0f} minutes**")
    st.write(f"Neural Network: **{main.nn.predict(order)[0]:.0f} minutes**")
