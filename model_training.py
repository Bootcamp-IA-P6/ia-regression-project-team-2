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
from features import preprocess_dataframe, FEATURES_LIST

print("Loading data...")
df = pd.read_csv('./data_madrid/newTest.csv')

cols_to_drop = ['Unnamed: 0','provincia','titulo','PrecioAnterior','planta','baños','tags','descripcion','Enlace']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df = df.dropna(subset=['PrecioActual'])
df = df[df['PrecioActual'] <= 6000000]

print("Processing variables...")
df_clean = preprocess_dataframe(df)

X = df_clean[FEATURES_LIST]
y = np.log1p(df_clean['PrecioActual'])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, FEATURES_LIST)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=10, cv=5, 
    scoring='r2', n_jobs=-1, random_state=42
)

print("Training GradientBoostingRegressor...")
search.fit(X_train, y_train)

model = search.best_estimator_

pred_log = model.predict(X_test)
pred_eur = np.expm1(pred_log)
y_test_eur = np.expm1(y_test)

model = search.best_estimator_

pred_test_log = model.predict(X_test)
pred_test_eur = np.expm1(pred_test_log)
y_test_eur = np.expm1(y_test)

pred_train_log = model.predict(X_train)

r2_train = r2_score(y_train, pred_train_log)
r2_test = r2_score(y_test, pred_test_log)
overfitting = r2_train - r2_test

print("-" * 30)
print(f"R2 Score (Test): {r2_test:.4f}")
print(f"MAE: {mean_absolute_error(y_test_eur, pred_test_eur):,.2f} €")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_eur, pred_test_eur)):,.2f} €")
print("-" * 30)

print("\n--- OVERFITTING ANALYSIS ---")
print(f"R2 Train: {r2_train:.4f}")
print(f"R2 Test:  {r2_test:.4f}")
print(f"Difference (Overfitting): {overfitting:.4f}")

if overfitting <= 0.05:
    print(f"Overfitting objective achieved {overfitting:.2%}")
else:
    print(f"Overfitting higher {overfitting:.2%}. Consider regular the model.")

print("\n--- Comparating mean prices (TEST) ---")
print(f"Real:     {y_test_eur.mean():,.2f} €")
print(f"Predicted: {pred_test_eur.mean():,.2f} €")
print("-" * 30)

joblib.dump(model, "house_price_model.pkl")
print("Model saved as 'house_price_model.pkl'")
