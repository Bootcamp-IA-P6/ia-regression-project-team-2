import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Madrid Real Estate Predictor")

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()


zonas_id_real = {
    "Salamanca": 1,
    "Retiro": 2,
    "Chamberí": 3,
    "Chamartín": 4,
    "Centro": 5,
    "Moncloa-Aravaca": 6,
    "Hortaleza": 7,
    "Fuencarral-El Pardo": 8,
    "Arganzuela": 9,
    "Ciudad Lineal": 10,
    "Tetuán": 11,
    "Barajas": 12,
    "San Blas-Canillejas": 13,
    "Moratalaz": 14,
    "Vicálvaro": 15,
    "Latina": 16,
    "Carabanchel": 17,
    "Puente de Vallecas": 18,
    "Usera": 19,
    "Villaverde": 20,
    "Villa de Vallecas": 21
}

zonas_prestigio = {
    "Salamanca": 6,
    "Retiro": 6,
    "Chamberí": 6,

    "Chamartín": 5,
    "Moncloa-Aravaca": 5,
    "Centro": 5,

    "Hortaleza": 4,
    "Fuencarral-El Pardo": 4,
    "Arganzuela": 4,

    "Ciudad Lineal": 3,
    "Tetuán": 3,
    "Barajas": 3,
    "San Blas-Canillejas": 3,

    "Moratalaz": 2,
    "Vicálvaro": 2,
    "Latina": 2,
    "Carabanchel": 2,

    "Puente de Vallecas": 1,
    "Usera": 1,
    "Villaverde": 1,
    "Villa de Vallecas": 1
}

prestigio_tipo = {
    "Estudio": 1,
    "Piso": 2,
    "Ático": 3,
    "Dúplex": 3,
    "Casa": 4,
    "Chalet": 4
}

st.title("🏠 Madrid House Price Predictor")

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        metros = st.number_input("Metros cuadrados", 20, 2000, 80)
        habitaciones = st.number_input("Habitaciones", 1, 15, 3)

        tipo_label = st.selectbox(
            "Tipo de vivienda",
            list(prestigio_tipo.keys())
        )
        tipo_prestigio = prestigio_tipo[tipo_label]

        zona_label = st.selectbox(
            "Distrito de Madrid",
            list(zonas_id_real.keys())
        )

        zona_id = zonas_id_real[zona_label]
        zona_prestigio = zonas_prestigio[zona_label]

    with col2:
        plantaN = st.number_input("Planta", 0, 50, 1)

        asc = st.selectbox("Ascensor", ["Si", "No"])
        ascensor = 1 if asc == "Si" else 0

        loc = st.selectbox("Localización", ["Exterior", "Interior"])
        es_exterior = 1 if loc == "Exterior" else 0

    submit = st.form_submit_button("💰 Calcular precio")

if submit:

    castigo_ascensor = -(plantaN * 2) if (plantaN > 3 and ascensor == 0) else 0

    input_df = pd.DataFrame([{
        'zona_id': zona_id,
        'zona_prestigio': zona_prestigio,
        'metros': metros,
        'habitaciones': habitaciones,
        'plantaN': plantaN,
        'ascensor': ascensor,
        'tipo_inmueble_prestigio': tipo_prestigio,
        'es_exterior': es_exterior,
        'castigo_ascensor': castigo_ascensor
    }])

    st.write("### 🔍 Datos enviados al modelo")
    st.dataframe(input_df)

    pred_log = model.predict(input_df)
    pred_eur = np.expm1(pred_log)[0]

    st.markdown("---")
    st.subheader(f"Precio estimado en {zona_label}")
    st.header(f"{pred_eur:,.2f} €")

    if castigo_ascensor < 0:
        st.warning(
            f"⚠️ Precio reducido: piso {plantaN} sin ascensor"
        )

    log_df = input_df.copy()
    log_df["predicted_price"] = pred_eur

    log_df.to_csv(
        "user_queries_log.csv",
        mode="a",
        header=not os.path.exists("user_queries_log.csv"),
        index=False
    )
