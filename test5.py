import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv('./data_madrid/newTest.csv')

cols_to_drop = [
    'Unnamed: 0','provincia','titulo','PrecioAnterior',
    'planta','baños','tags','descripcion','Enlace'
]

df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

zonas_prestigio_madrid = {
    "Salamanca": 6,
    "Chamberí": 6,
    "Retiro": 6,
    "Chamartín": 5,
    "Centro": 5,
    "Moncloa-Aravaca": 5,
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
    "Usera": 1,
    "Villaverde": 1,
    "Puente de Vallecas": 1,
    "Villa de Vallecas": 1
}


if "zona_id" in df_clean.columns:
    df_clean["zona_prestigio"] = (
        df_clean["zona_id"]
        .map(zonas_prestigio_madrid)
        .fillna(3)
    )
else:
    df_clean["zona_prestigio"] = 3

prestigio_tipo = {
    'estudio': 1,
    'piso': 2,
    'atico': 3,
    'duplex': 3,
    'casa': 4,
    'chalet': 4
}

df_clean["tipo_inmueble_prestigio"] = (
    df_clean["tipo_inmueble_id"]
    .astype(str)
    .str.lower()
    .map(prestigio_tipo)
    .fillna(2)
)

df_clean['es_exterior'] = df_clean['localizacion'].apply(
    lambda x: 1 if str(x).lower() == 'exterior' else 0
)

df_clean['plantaN'] = pd.to_numeric(df_clean['plantaN'], errors='coerce').fillna(0)
df_clean['ascensor'] = pd.to_numeric(df_clean['ascensor'], errors='coerce').fillna(0)

df_clean['castigo_ascensor'] = np.where(
    (df_clean['plantaN'] > 3) & (df_clean['ascensor'] == 0),
    -(df_clean['plantaN'] * 2),
    0
)

df_clean = df_clean.dropna(subset=['PrecioActual'])
df_clean = df_clean[df_clean['PrecioActual'] <= 6000000]

features = [
    'zona_id',
    'zona_prestigio',
    'metros',
    'habitaciones',
    'plantaN',
    'ascensor',
    'tipo_inmueble_prestigio',
    'es_exterior',
    'castigo_ascensor'
]

X = df_clean[features]
y = np.log1p(df_clean['PrecioActual'])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

param_dist = {
    'regressor__n_estimators': [200, 400, 600],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.8, 0.9, 1.0]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=10,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

print("Training model...")
search.fit(X_train, y_train)

model = search.best_estimator_

pred_log = model.predict(X_test)
pred_eur = np.expm1(pred_log)
y_test_eur = np.expm1(y_test)

print("R2:", r2_score(y_test, pred_log))
print("MAE €:", mean_absolute_error(y_test_eur, pred_eur))
print("RMSE €:", np.sqrt(mean_squared_error(y_test_eur, pred_eur)))

joblib.dump(model, "house_price_model.pkl")
print("Model saved.")
