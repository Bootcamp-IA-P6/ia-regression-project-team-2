import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer 
# from features import HouseFeatureAdder

# LOAD DATASET
df = pd.read_csv('./data_madrid/newTest.csv')

precioCasa = df[df['PrecioActual'] > 6000000]['PrecioActual'].count()
print(precioCasa)

# FEATURE ENGINEERING (Before splitting X and y) // Eliminamos columnas que no aportan valor(RUIDO)
cols_to_drop = ['Unnamed: 0','provincia', 'titulo', 'PrecioAnterior', 'planta', 'baños', 'tags', 'descripcion', 'Enlace', 'zona']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Create 'accessibility_index' / He creado una nueva variable, que ayudaría al modelo a entender el porqué de la diferencia de precio entre pisos altos y bajos. (Esto nos daría una presición aún mayor)
if 'plantaN' in df_clean.columns and 'ascensor' in df_clean.columns:
    # Ensure they are numeric
    df_clean['plantaN'] = pd.to_numeric(df_clean.get('plantaN', 0), errors='coerce').fillna(0)
    df_clean['ascensor'] = pd.to_numeric(df_clean.get('ascensor', 0), errors='coerce').fillna(0)

# Lógica de castigo y premio
# if 'plantaN' in df_clean.columns and 'ascensor' in df_clean.columns:
#     df_clean['castigo_ascensor'] = np.where(
#         (df_clean['plantaN'] > 3) & (df_clean['ascensor'] == 0), 
#         df_clean['plantaN'] * 2, 
#         0
#     )

# if 'tipo_inmueble' in df_clean.columns:
#     df_clean['es_exterior'] = df_clean['tipo_inmueble'].apply(lambda x: 1 if str(x).lower() == 'exterior' else 0)
# else:
#     df_clean['es_exterior'] = 0

# Clean target
df_clean = df_clean.dropna(subset=['PrecioActual'])
df_clean = df_clean[df_clean['PrecioActual'] <= 6000000]

# Define Features (X) and Target (y)
X = df_clean.drop('PrecioActual', axis=1)
# Apply Log transformation to handle price variance
y = np.log1p(df_clean['PrecioActual'])  # Aplico el logaritmo, con esto reducimos la brecha entre precios, para evitar errores entre casas muy costosas o muy economicas.(Para que el modelo puedea aprender de un mercado generalizado)

# PREPROCESSING PIPELINE / Si el modelo encuentra algún valor nulo, lo sustituye por la media de la columna. (Esto ayuda a que todos tengan el mismo peso y evitar que el modelo se equivoque)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Add new features to numeric_features
# new_features = ['accessibility_index', 'rooms_per_meter', 'avg_room_size']
# numeric_features.extend(new_features) 

# Pipeline para columnas numéricas / (imputer: Si el modelo encuentra algún valor nulo, lo sustituye por la media de la columna.
# scaler: Estandariza los datos, para que todos tengan el mismo peso y evitar que el modelo se equivoque)
numeric_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
 
# Orquestador, que decide que hacer con cada columna, ya sea un entero, un decimal o texto.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Une todo en un solo pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# MODEL PIPELINE / Aquí se une todo en un solo pipeline, el preprocesador y el modelo. 
# (con esto conseguimos que el modelo sea más eficiente y preciso, al usarse una vez,
# no se tendrá que volver a escribir el código de limpieza, el modelo ya sabrá limpiarse solo.)
full_pipeline = Pipeline(steps=[
#    ('feature_generation', HouseFeatureAdder()),
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Training and optimization. / Parametros que el modelo utilizará para aprender.
# (n_estimators: Número de árboles a utilizar.
# learning_rate: Tasa de aprendizaje.
# max_depth: Profundidad máxima de los árboles.
# subsample: Proporción de muestras a utilizar en cada árbol.)
param_distributions = {
    'regressor__n_estimators': [100, 300, 500],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 6, 8],  
    'regressor__subsample': [0.8, 0.9, 1.0]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if 'plantaN' in X_train.columns and 'ascensor' in X_train.columns:
    X_train['castigo_ascensor'] = np.where(
        (X_train['plantaN'] > 3) & (X_train['ascensor'] == 0), 
        X_train['plantaN'] * 2, 
        0
    )

if 'tipo_inmueble' in X_train.columns:
    X_train['es_exterior'] = X_train['tipo_inmueble'].apply(lambda x: 1 if str(x).lower() == 'exterior' else 0)
else:
    X_train['es_exterior'] = 0

# RandomizedSearchCV: Busca los mejores hiperparámetros para el modelo.
# n_iter: Número de combinaciones a probar.
# cv: Número de pliegues para la validación cruzada.
# scoring: Métrica a optimizar (R² en este caso).
# n_jobs: Número de núcleos a utilizar (-1 significa todos).
# random_state: Semilla para reproducibilidad.
search = RandomizedSearchCV(full_pipeline, param_distributions, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)

print("Training model (Log-Price)...")
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Evaluation (Back-transforming from Log to Euros) / Evaluación (Volviendo a convertir los datos de logaritmo a euros)
train_preds_log = best_model.predict(X_train)
test_preds_log = best_model.predict(X_test)


# Convert back to Euros using np.expm1 / Vuelvo a convertir los datos a euros para poder compararlos. (Ya que al aplicar el logaritmo, los datos se distorsionan.)
y_test_euros = np.expm1(y_test)
test_preds_euros = np.expm1(test_preds_log)

# Métricas de evaluación
# R²: Mide la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. (Precisión del modelo)
# Difference: Mide la diferencia entre el R² de entrenamiento y el R² de prueba.
# MAE: Mide la diferencia promedio entre los valores reales y los valores predichos. (Error promedio)
# RMSE: Mide la desviación estándar de los residuos (la diferencia entre los valores reales y los valores predichos -- Penalización de errores graves)

train_r2 = r2_score(y_train, train_preds_log)
test_r2 = r2_score(y_test, test_preds_log)
mae_euros = mean_absolute_error(y_test_euros, test_preds_euros)
rmse_euros = np.sqrt(mean_squared_error(y_test_euros, test_preds_euros))

print("-" * 30)
print(f"R² Train: {train_r2:.4f}")
print(f"R² Test: {test_r2:.4f}")
print(f"Difference: {abs(train_r2 - test_r2)*100:.2f}%")
print(f"MAE (Euros): {mae_euros:,.2f} €")
print(f"RMSE (Euros): {rmse_euros:,.2f} €")
print("-" * 30)

if abs(train_r2 - test_r2) < 0.05:
    print("SUCCESS: Overfitting below 5%.")
else:
    print("WARNING: Overfitting detected.")

joblib.dump(best_model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")
