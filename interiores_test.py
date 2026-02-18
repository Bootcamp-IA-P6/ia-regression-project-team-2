import pandas as pd
from features import preprocess_dataframe

df = pd.read_csv('./data_madrid/newTest.csv')

print("--- uniques values of 'localizacion' ---")
print(df['localizacion'].unique())

print("\n--- count of rows by value ---")
print(df['localizacion'].value_counts(dropna=False))
df_clean = preprocess_dataframe(df)

print("\n--- final result (es_exterior) ---")
conteo = df_clean['es_exterior'].value_counts()
porcentaje = df_clean['es_exterior'].value_counts(normalize=True) * 100

print(f"Exterior (1): {conteo.get(1, 0)} inmuebles ({porcentaje.get(1, 0):.2f}%)")
print(f"Interior (0): {conteo.get(0, 0)} inmuebles ({porcentaje.get(0, 0):.2f}%)")

print("\n--- mean price by es_exterior ---")
print(df_clean.groupby('es_exterior')['PrecioActual'].mean().round(2))
