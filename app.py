import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# Configuración página
# ==========================
st.set_page_config(
    page_title="Predicción Inmobiliaria",
    page_icon="🏠",
    layout="centered"
)

# ==========================
# Cargar modelo
# ==========================
@st.cache_resource
def cargar_modelo():
    ruta = Path("mi_modelo_entrenado.joblib")
    if not ruta.exists():
        st.error("No se encontró el archivo mi_modelo_entrenado.joblib")
        st.stop()
    modelo = joblib.load(ruta)
    return modelo

modelo = cargar_modelo()

# ==========================
# Interfaz
# ==========================
st.title("Predicción de Precio de Vivienda")

st.markdown("Introduce las características:")

metros = st.number_input("Metros cuadrados", min_value=10, step=1)
habitaciones = st.number_input("Habitaciones", min_value=0.0, step=0.5)
ascensor = st.selectbox("Ascensor", [0, 1])
localizacion = st.number_input("Localización (codificada)", min_value=0, step=1)
plantaN = st.number_input("Número de planta", min_value=0.0, step=1.0)
zona_id = st.number_input("Zona ID", min_value=0, step=1)
tipo_inmueble_id = st.number_input("Tipo inmueble ID", min_value=0, step=1)

# ==========================
# Predicción
# ==========================
if st.button("Predecir precio"):

    datos = np.array([[
        int(metros),
        float(habitaciones),
        int(ascensor),
        int(localizacion),
        float(plantaN),
        int(zona_id),
        int(tipo_inmueble_id)
    ]])

    try:
        prediccion = modelo.predict(datos)[0]

        st.success(f"Precio estimado: {prediccion:,.2f} €")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")

