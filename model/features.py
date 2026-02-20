import pandas as pd
import numpy as np


ZONAS_DATA = {
    "Salamanca": {"id": 1, "prestigio": 6},
    "Retiro": {"id": 2, "prestigio": 6},
    "Chamberí": {"id": 3, "prestigio": 6},
    "Chamartín": {"id": 4, "prestigio": 5},
    "Centro": {"id": 5, "prestigio": 5},
    "Moncloa-Aravaca": {"id": 6, "prestigio": 5},
    "Hortaleza": {"id": 7, "prestigio": 4},
    "Fuencarral-El Pardo": {"id": 8, "prestigio": 4},
    "Arganzuela": {"id": 9, "prestigio": 4},
    "Ciudad Lineal": {"id": 10, "prestigio": 3},
    "Tetuán": {"id": 11, "prestigio": 3},
    "Barajas": {"id": 12, "prestigio": 3},
    "San Blas-Canillejas": {"id": 13, "prestigio": 3},
    "Moratalaz": {"id": 14, "prestigio": 2},
    "Vicálvaro": {"id": 15, "prestigio": 2},
    "Latina": {"id": 16, "prestigio": 2},
    "Carabanchel": {"id": 17, "prestigio": 2},
    "Puente de Vallecas": {"id": 18, "prestigio": 1},
    "Usera": {"id": 19, "prestigio": 1},
    "Villaverde": {"id": 20, "prestigio": 1},
    "Villa de Vallecas": {"id": 21, "prestigio": 1}
}

PRESTIGIO_TIPO = {
    'estudio': 1, 'piso': 2, 'atico': 3, 'duplex': 3, 'casa': 4, 'chalet': 4,
    '1': 1, '2': 2, '3': 3, '4': 3, '5': 4, '6': 4 
}

FEATURES_LIST = [
    'zona_id', 'zona_prestigio', 'metros', 'habitaciones', 
    'plantaN', 'ascensor', 'tipo_inmueble_prestigio', 
    'es_exterior', 'castigo_ascensor'
]

def calcular_castigo_ascensor(planta, tiene_ascensor):
    return -(planta * 2) if (planta > 3 and tiene_ascensor == 0) else 0

def preprocess_dataframe(df):
    df = df.copy()
    
    map_prestigio = {v['id']: v['prestigio'] for k, v in ZONAS_DATA.items()}
    df['zona_prestigio'] = df['zona_id'].map(map_prestigio).fillna(3)
    
    df["tipo_inmueble_prestigio"] = (
        df["tipo_inmueble_id"].astype(str)
        .str.replace('.0', '', regex=False)
        .str.lower()
        .str.strip()
        .map(PRESTIGIO_TIPO)
        .fillna(2)
    )
    
    def limpiar_localizacion(x):
        val = str(x).strip().lower().replace('.0', '')
        if val in ['1', 'exterior']:
            return 1
        return 0
    df['es_exterior'] = df['localizacion'].apply(limpiar_localizacion)
    
    df['plantaN'] = pd.to_numeric(df['plantaN'], errors='coerce').fillna(0)
    df['ascensor'] = pd.to_numeric(df['ascensor'], errors='coerce').fillna(0)
    
    df['castigo_ascensor'] = df.apply(
        lambda x: calcular_castigo_ascensor(x['plantaN'], x['ascensor']), axis=1
    )
    
    return df
