import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from features import ZONAS_DATA, PRESTIGIO_TIPO, calcular_castigo_ascensor, FEATURES_LIST

st.set_page_config(page_title="Madrid Real Estate Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

try:
    model = load_model()
except:
    st.error("Error finding 'house_price_model.pkl'. Run train.py first.")
    st.stop()

st.title("🏠 Madrid House Price Predictor")
st.markdown("Calculate the market price based on historical district data.")

with st.form("form_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        metros = st.number_input("Metros cuadrados", 20, 2000, 80)
        habitaciones = st.number_input("Habitaciones", 1, 15, 3)
        
        tipo_label = st.selectbox(
            "Tipo de vivienda",
            options=[k.capitalize() for k in PRESTIGIO_TIPO.keys()]
        )
        
        zona_label = st.selectbox(
            "Distrito de Madrid",
            options=list(ZONAS_DATA.keys())
        )

    with col2:
        plantaN = st.number_input("Planta", 0, 50, 1)
        asc = st.selectbox("Ascensor", ["Si", "No"])
        ascensor = 1 if asc == "Si" else 0
        
        loc = st.selectbox("Localización", ["Exterior", "Interior"])
        es_exterior = 1 if loc == "Exterior" else 0

    submit = st.form_submit_button("💰 Calculate estimated price")

if submit:
    zona_id = ZONAS_DATA[zona_label]['id']
    zona_prestigio = ZONAS_DATA[zona_label]['prestigio']
    tipo_prestigio = PRESTIGIO_TIPO[tipo_label.lower()]
    castigo = calcular_castigo_ascensor(plantaN, ascensor)

    input_df = pd.DataFrame([{
        'zona_id': zona_id,
        'zona_prestigio': zona_prestigio,
        'metros': metros,
        'habitaciones': habitaciones,
        'plantaN': plantaN,
        'ascensor': ascensor,
        'tipo_inmueble_prestigio': tipo_prestigio,
        'es_exterior': es_exterior,
        'castigo_ascensor': castigo
    }])

    input_df = input_df[FEATURES_LIST]

    pred_log = model.predict(input_df)
    pred_eur = np.expm1(pred_log)[0]

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Estimated price for {zona_label}")
        st.header(f"{pred_eur:,.2f} €")
    with c2:
        st.info(f"Price per m²: {pred_eur/metros:,.2f} €/m²")

    if castigo < 0:
        st.warning(f"The price has a negative adjustment for being a {plantaN} without an elevator.")

    log_file = "user_queries_log.csv"
    input_df["predicted_price"] = pred_eur
    input_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
