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

st.title("🏠 Madrid House Price Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        metros = st.number_input("Metros cuadrados (m²)", min_value=15, max_value=2000, value=80)
        habitaciones = st.number_input("Habitaciones", min_value=1, max_value=20, value=3)
        loc_option = st.selectbox("Tipo de Vivienda", options=["Exterior", "Interior"])
        es_exterior = 1 if loc_option == "Exterior" else 0
        localizacion = es_exterior 
        
    with col2:
        plantaN = st.number_input("Planta", min_value=0, max_value=50, value=1)
        asc_option = st.selectbox("¿Tiene ascensor?", options=["Si", "No"])
        ascensor = 1 if asc_option == "Si" else 0
        tipo_inmueble_id = 1 

    submit = st.form_submit_button("💰 Calculate Price")

if submit:
    # LÓGICA DE CASTIGO
    castigo_ascensor = plantaN * 2 if (plantaN > 3 and ascensor == 0) else 0

    input_data = pd.DataFrame([{
        'zona_id': 1,    
        'metros': float(metros),
        'habitaciones': int(habitaciones),
        'localizacion': int(localizacion),
        'plantaN': int(plantaN),
        'ascensor': int(ascensor),
        'tipo_inmueble_id': int(tipo_inmueble_id),
        'castigo_ascensor': float(castigo_ascensor),
        'es_exterior': int(es_exterior)
    }])
    
    st.write("🔍 **Data Monitoring**")
    st.write("Data sent to the model:")
    st.dataframe(input_data)
    
    try:
        prediction_log = model.predict(input_data)
        prediction_euros = np.expm1(prediction_log)[0]
        
        st.markdown("---")
        st.subheader(f"Estimated price:")
        st.header(f"{prediction_euros:,.2f} €")
        
        # Alerta visual de castigo si aplica
        if castigo_ascensor > 0:
            st.warning(f"⚠️ El precio ha sido penalizado por ser un {plantaN}º piso sin ascensor.")
        
        query_save = input_data.copy()
        query_save['predicted_price'] = prediction_euros
        
        file_path = 'user_queries_log.csv'
        header = not os.path.exists(file_path)
        query_save.to_csv(file_path, mode='a', header=header, index=False)
        
        st.info("Consulting saved in the monitoring system.")

    except Exception as e:
        st.error(f"Error in the prediction: {e}")
