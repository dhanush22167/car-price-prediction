"""
app.py
------
Streamlit front-end for the Used Car Price Predictor.
Loads the trained pipeline (models/model.pkl) and the saved
dropdown/range metadata (models/options.json) to build the input form,
then returns a predicted price.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    with open("models/options.json") as f:
        options = json.load(f)
    metrics = {}
    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return model, options, metrics


model, options, metrics = load_artifacts()
num_ranges = options["numeric_ranges"]
cat_options = options["categorical_options"]

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the resale value of a car based on its specifications. "
    "Trained on ~11,900 real used-car listings using **XGBoost**."
)

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", cat_options["Make"], index=cat_options["Make"].index("Toyota") if "Toyota" in cat_options["Make"] else 0)
        year = st.slider("Year", int(num_ranges["Year"]["min"]), int(num_ranges["Year"]["max"]), int(num_ranges["Year"]["median"]))
        engine_hp = st.slider("Engine HP", int(num_ranges["Engine HP"]["min"]), int(num_ranges["Engine HP"]["max"]), int(num_ranges["Engine HP"]["median"]))
        engine_cyl = st.selectbox("Engine Cylinders", sorted(set(int(x) for x in [0, 3, 4, 5, 6, 8, 10, 12])), index=3)
        doors = st.selectbox("Number of Doors", [2, 3, 4], index=2)

    with col2:
        transmission = st.selectbox("Transmission Type", cat_options["Transmission Type"])
        drive = st.selectbox("Driven Wheels", cat_options["Driven_Wheels"])
        size = st.selectbox("Vehicle Size", cat_options["Vehicle Size"])
        style = st.selectbox("Vehicle Style", cat_options["Vehicle Style"])
        highway_mpg = st.slider("Highway MPG", 12, 60, 28)
        city_mpg = st.slider("City MPG", 9, 50, 20)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "Year": year,
        "Engine HP": engine_hp,
        "Engine Cylinders": engine_cyl,
        "Number of Doors": doors,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Make": make,
        "Transmission Type": transmission,
        "Driven_Wheels": drive,
        "Vehicle Size": size,
        "Vehicle Style": style,
    }])

    pred_log = model.predict(input_df)[0]
    pred_price = float(np.expm1(pred_log))

    st.success(f"### Estimated Price: ${pred_price:,.0f}")
    st.caption(
        "This is an estimate based on historical listings, not an exact valuation."
    )

st.divider()
if metrics:
    best = options.get("best_model", "")
    if best in metrics:
        st.caption(
            f"Model: **{best}**  |  Test RMSE: **${metrics[best]['rmse']:,.0f}**  |  "
            f"Test R²: **{metrics[best]['r2']:.3f}**"
        )
st.caption("Built with scikit-learn, XGBoost & Streamlit · [GitHub repo link here]")
