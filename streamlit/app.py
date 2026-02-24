import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from model.features import ZONAS_DATA, PRESTIGIO_TIPO,  calcular_castigo_ascensor, FEATURES_LIST



st.set_page_config(page_title="Sistema de precios de Viviendas — Madrid", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(root_path, 'model', 'house_price_model.pkl')
    return joblib.load(model_path)
try:
    model = load_model()
except:
    st.error("Error finding 'house_price_model.pkl'. Run model_training.py first.")
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.markdown(r"""
    Trained on **11,826 listings** from Idealista Madrid.

    **Algorithm:** Gradient Boosting Regressor
    
    **R² score:** 0.91
    
    **Mean error:** ±180,000 €

    **Districts covered:** 21 districts of Madrid
    
    **Price range:** up to 6,000,000 €
    """)
    st.markdown("---")
    st.caption("Data source: Idealista Madrid (Kaggle)")

st.title("🏠 Predictor de Precios de Vivienda en Madrid")
st.markdown("Calcula el precio estimado de mercado basado en datos históricos de los distritos de Madrid.")
st.markdown("---")

with st.form("form_prediccion"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(r"**📐 Tamaño**")
        metros = st.number_input("Superficie (m²)", 20, 2000, 60)
        habitaciones = st.number_input("Habitaciones", 1, 15, 2)

    with col2:
        st.markdown(r"**🏘️ Inmueble**")
        tipo_label = st.selectbox(
            "Tipo de inmueble",
            options=["Estudio", "Piso", "Ático", "Dúplex", "Casa", "Chalet"]
        )
        zona_label = st.selectbox(
            "Distrito",
            options=list(ZONAS_DATA.keys())
        )

    with col3:
        st.markdown(r"**🏢 Edificio**")
        plantaN = st.number_input("Planta", 0, 50, 1, help="Planta 0 = planta baja")
        asc = st.selectbox("Ascensor", ["Yes", "No"], help="Afecta al precio a partir de la planta 3")
        loc = st.selectbox("Localizacion", ["Exterior", "Interior"])

    ascensor = 1 if (asc or "No") == "Yes" else 0
    es_exterior = 1 if (loc or "Exterior") == "Exterior" else 0

    submit = st.form_submit_button("💰 Calcular precio estimado")

if submit:
    zona_id = ZONAS_DATA[zona_label or "Salamanca"]['id']
    zona_prestigio = ZONAS_DATA[zona_label or "Salamanca"]['prestigio']
    TIPO_A_PRESTIGIO = {"Estudio": 1, "Piso": 2, "Ático": 3, "Dúplex": 3, "Casa": 4, "Chalet": 4}
    tipo_prestigio = TIPO_A_PRESTIGIO[tipo_label or "Piso"]
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
        st.metric(
            label=f"Precio estimado en {zona_label}",
            value=f"{pred_eur:,.0f} €"
        )
    with c2:
        st.metric(
            label="Precio por m²",
            value=f"{pred_eur/metros:,.0f} €/m²"
        )

    if castigo < 0:
        st.warning(f"⚠️ Precio ajustado a la baja: planta {int(plantaN)} sin ascensor.")

    log_file = "results_queries/user_queries_log.csv"
    input_df["predicted_price"] = pred_eur
    input_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
