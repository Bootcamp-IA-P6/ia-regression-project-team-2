# Encoding de Variables Categóricas para el Modelo

**Fecha:** 2026-02-19

## Contexto: qué problema estamos resolviendo

El notebook `dataClean.ipynb` transforma las variables categóricas antes de entrenar, pero con una técnica que introduce ruido en el modelo. El resultado es un R² de ~0.68, que es mediocre para este dataset.

El problema está en cómo se codifican `zona`, `tipo_inmueble`, `ascensor` y `localizacion`.

---

## El problema: Label Encoding en variables nominales

### ¿Qué es Label Encoding?

Es lo que se hace actualmente: asignar un número entero a cada categoría.

```python
# Lo que hace el notebook ahora:
df_clean['zona_id'] = df_clean['zona'].astype('category').cat.codes + 1
# arganzuela=1, barajas=2, barrio-de-salamanca=3, carabanchel=4...

df_clean['tipo_inmueble_id'] = df_clean['tipo_inmueble'].astype('category').cat.codes + 1
# Casa=1, Chalet=2, Dúplex=3, Estudio=4, Piso=5, Ático=6
```

### ¿Por qué es un problema?

Un modelo de regresión lineal trata todos los números como **números de verdad**, con orden y distancia entre ellos. Al asignar IDs, le estás diciendo implícitamente al modelo cosas falsas:

- Que `barrio-de-salamanca (3)` está a la misma "distancia" de `barajas (2)` que de `carabanchel (4)`
- Que `Chalet (2)` vale "el doble" que `Casa (1)` en algún sentido
- Que la zona `tetuan (17)` es 17 veces más que `arganzuela (1)`

El modelo intentará encontrar una relación lineal entre esos IDs y el precio, pero el orden es **puramente alfabético**, no tiene ninguna relación con los precios reales.

### El problema adicional del -1 para NO_APLICA

```python
# Ascensor: S=1, N=0, NO_APLICA=-1
# Localización: EXTERIOR=1, INTERIOR=0, NO_APLICA=-1
```

El -1 corresponde a chalets y casas, que son los inmuebles **más caros** del dataset. El modelo aprenderá que "ascensor = -1" predice precios altos, cuando en realidad lo que predice precios altos es que sea un chalet. Estás mezclando dos señales en una sola columna.

### El problema del plantaN=0 para NO_APLICA

Los chalets tienen `plantaN=0`, igual que los pisos en planta baja. El modelo no puede distinguir entre un piso bajo (generalmente más barato) y un chalet sin planta (generalmente muy caro). Las dos señales se anulan.

---

## Las soluciones correctas

### Variable: `zona` → Target Encoding

**¿Qué es?** Sustituir cada zona por la **media del precio** de esa zona, calculada solo con los datos de entrenamiento.

**¿Por qué?** Es la técnica más adecuada para variables nominales con muchas categorías en problemas de regresión. En vez de decirle al modelo "esta zona es la 3", le dices "en esta zona los pisos cuestan de media 1.2M€". Eso sí tiene información real.

**⚠️ Truco importante:** La media se calcula SOLO sobre los datos de train, y luego se aplica (como un mapping) a test. Si la calculas sobre todo el dataset, introduces data leakage.

```python
# --- PASO 1: Separar train y test ANTES de hacer el encoding ---
from sklearn.model_selection import train_test_split

X = df.drop(columns=['PrecioActual'])
y = df['PrecioActual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASO 2: Calcular la media de precio por zona en TRAIN ---
# Necesitamos juntar X_train con y_train temporalmente para calcular la media
train_con_precio = X_train.copy()
train_con_precio['PrecioActual'] = y_train

media_precio_por_zona = train_con_precio.groupby('zona')['PrecioActual'].mean()
# Resultado: {'arganzuela': 450000, 'barrio-de-salamanca': 1800000, ...}

# --- PASO 3: Aplicar el mapping a train y test ---
X_train['zona_encoded'] = X_train['zona'].map(media_precio_por_zona)
X_test['zona_encoded'] = X_test['zona'].map(media_precio_por_zona)

# --- PASO 4: Eliminar la columna original de texto ---
X_train = X_train.drop(columns=['zona'])
X_test = X_test.drop(columns=['zona'])
```

---

### Variable: `tipo_inmueble` → One-Hot Encoding

**¿Qué es?** Crear una columna binaria (0 o 1) por cada categoría.

**¿Por qué?** Con 6 categorías es manejable, y evita el problema del orden artificial. El modelo aprende el coeficiente de cada tipo de forma independiente.

```python
# get_dummies crea columnas binarias automáticamente
# drop_first=True elimina una columna para evitar multicolinealidad perfecta
# (si tienes 6 tipos, con 5 columnas puedes inferir el 6º)

X_train = pd.get_dummies(X_train, columns=['tipo_inmueble'], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=['tipo_inmueble'], drop_first=True)

# ⚠️ Alinear columnas: si en test no aparece algún tipo, puede faltar una columna
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Resultado: columnas como tipo_inmueble_Chalet, tipo_inmueble_Dúplex, etc.
# Cada una vale 1 si el inmueble es de ese tipo, 0 si no.
```

---

### Variables: `ascensor` y `localizacion` → One-Hot Encoding de los 3 valores

El problema actual es meter S/N/NO_APLICA en una sola columna numérica. Lo correcto es tratar los tres valores como categorías independientes.

```python
# Volvemos a los valores de texto originales (antes de convertir a 1/0/-1)
# Hacemos el encoding sobre la columna con 'S', 'N', 'NO_APLICA'

for col in ['ascensor', 'localizacion']:
    dummies = pd.get_dummies(X_train[col], prefix=col, drop_first=False)
    # Resultado: ascensor_N, ascensor_NO_APLICA, ascensor_S

    # Quitamos la columna original y añadimos las dummies
    X_train = pd.concat([X_train.drop(columns=[col]), dummies], axis=1)

# Lo mismo para test (y alinear columnas después)
for col in ['ascensor', 'localizacion']:
    dummies = pd.get_dummies(X_test[col], prefix=col, drop_first=False)
    X_test = pd.concat([X_test.drop(columns=[col]), dummies], axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# El problema es que train y test son muestras distintas del dataset. Es
#   perfectamente posible que, por ejemplo, en tu test no haya ningún inmueble con
#    ascensor=NO_APLICA (por azar del split). En ese caso:

#   - X_train tendrá las columnas: ascensor_N, ascensor_NO_APLICA, ascensor_S
#   - X_test solo tendrá: ascensor_N, ascensor_S

#   Cuando intentes escalar y predecir, el modelo espera las mismas columnas con
#   las que entrenó — si falta una, peta.

#   reindex fuerza a que X_test tenga exactamente las mismas columnas que X_train,
#    en el mismo orden. Si falta alguna, la crea con fill_value=0 (ceros, que en
#   one-hot significa "esa categoría no aparece").

#   Se hace fuera del bucle porque solo necesitas hacerlo una vez, después de
#   haber procesado todas las columnas. Si lo hicieras dentro del bucle después de
#    cada columna, estarías reindexando con un X_train incompleto (al que todavía
#   le faltan las dummies de localizacion).

```

---

### Variable: `plantaN` — Separar el NO_APLICA

En vez de poner 0 a los chalets (que confunde "planta baja" con "no tiene planta"), crear una columna separada que indica si el inmueble tiene planta o no.

```python
# Crear columna binaria: 1 si tiene planta, 0 si es chalet/casa
X_train['tiene_planta'] = (X_train['plantaN'] != 'NO_APLICA').astype(int)
X_test['tiene_planta']  = (X_test['plantaN']  != 'NO_APLICA').astype(int)

# Para los NO_APLICA, ponemos NaN (que luego imputamos con la mediana de pisos)
X_train['plantaN'] = pd.to_numeric(X_train['plantaN'].replace('NO_APLICA', float('nan')))
X_test['plantaN']  = pd.to_numeric(X_test['plantaN'].replace('NO_APLICA',  float('nan')))

# Imputar con la mediana de pisos (calculada sobre train)
mediana_planta_train = X_train.loc[X_train['tiene_planta'] == 1, 'plantaN'].median()
X_train['plantaN'] = X_train['plantaN'].fillna(mediana_planta_train)
X_test['plantaN']  = X_test['plantaN'].fillna(mediana_planta_train)


# El problema que resuelve es este: si un chalet tiene plantaN=0 y un piso en
#   planta baja también tiene plantaN=0, el modelo los ve igual. Pero son cosas
#   muy distintas — un chalet suele costar mucho más que un piso bajo.

#   La solución divide esa información en dos señales separadas:

#   tiene_planta — le dice al modelo si el concepto de "planta" aplica o no:
#   Piso 3ª       → tiene_planta = 1
#   Piso bajo     → tiene_planta = 1
#   Chalet        → tiene_planta = 0
#   Casa          → tiene_planta = 0

#   plantaN — el número de planta real, solo para los que sí tienen:
#   Piso 3ª       → plantaN = 3
#   Piso bajo     → plantaN = 0
#   Chalet        → plantaN = ??? (no tiene sentido)
#   Casa          → plantaN = ??? (no tiene sentido)

#   Para los chalets/casas, plantaN no tiene valor real, así que se les pone la
#   mediana de los pisos (un número neutro). No es perfecto, pero ya da igual
#   porque tiene_planta=0 ya le avisa al modelo de que ese dato no es fiable.

#   Ahora el modelo tiene dos señales limpias:
#   - "¿Es un unifamiliar (chalet/casa)?" → tiene_planta
#   - "¿En qué planta está?" → plantaN (solo relevante cuando tiene_planta=1)

```

---

## Cómo quedaría el flujo completo en el notebook de modelo

Este es el flujo que reemplazaría todo lo que hay después de leer el CSV, antes del StandardScaler:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# --- Cargar datos limpios (sin los encodings numéricos del dataClean) ---
# IMPORTANTE: esto asume que dataClean guarda las columnas de texto (zona, tipo_inmueble,
# ascensor, localizacion) y NO las versiones numéricas (zona_id, tipo_inmueble_id, etc.)
# Si el CSV ya tiene las versiones numéricas, hay que modificar el dataClean primero.

df = pd.read_csv('./data_madrid/datos_limpios.csv')

X = df.drop(columns=['PrecioActual'])
y = df['PrecioActual']

# --- Split primero ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Target Encoding para zona ---
train_tmp = X_train.copy()
train_tmp['PrecioActual'] = y_train.values
media_zona = train_tmp.groupby('zona')['PrecioActual'].mean()

X_train['zona_encoded'] = X_train['zona'].map(media_zona)
X_test['zona_encoded']  = X_test['zona'].map(media_zona)

# Si alguna zona de test no aparece en train (raro pero posible), usar la media global
media_global = y_train.mean()
X_train['zona_encoded'] = X_train['zona_encoded'].fillna(media_global)
X_test['zona_encoded']  = X_test['zona_encoded'].fillna(media_global)

X_train = X_train.drop(columns=['zona'])
X_test  = X_test.drop(columns=['zona'])

# --- One-Hot Encoding para tipo_inmueble ---
X_train = pd.get_dummies(X_train, columns=['tipo_inmueble'], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=['tipo_inmueble'], drop_first=True)

# --- One-Hot Encoding para ascensor y localizacion ---
for col in ['ascensor', 'localizacion']:
    dummies_train = pd.get_dummies(X_train[col], prefix=col)
    dummies_test  = pd.get_dummies(X_test[col],  prefix=col)

    X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
    X_test  = pd.concat([X_test.drop(columns=[col]),  dummies_test],  axis=1)

# Alinear columnas train/test (puede haber diferencias si algún valor no aparece en test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- Escalar ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_train.columns)
```

---

## ¿Qué impacto debería tener esto?

- **zona_encoded** con target encoding debería ser la feature más importante del modelo. Un piso en barrio-de-salamanca vale ~4x más que uno en villaverde, y ahora el modelo tiene esa información directamente.
- **tipo_inmueble** con one-hot deja que el modelo aprenda un precio base diferente para cada tipo sin imponer un orden falso.
- **ascensor/localizacion** con one-hot separa correctamente el "no aplica" de los chalets de los valores reales de pisos.

El R² de 0.68 debería mejorar notablemente, especialmente gracias al target encoding de zona.

---

## Consideración importante antes de implementar

El CSV `datos_limpios.csv` que genera el `dataClean.ipynb` **ya tiene los encodings numéricos** (zona_id, tipo_inmueble_id, ascensor=1/0/-1, etc.) y ha **eliminado las columnas de texto originales**. Para aplicar este encoding correctamente, hay dos opciones:

1. **Modificar el dataClean** para que guarde las columnas de texto originales (zona, tipo_inmueble, ascensor, localizacion como 'S'/'N'/'NO_APLICA') en el CSV
2. **Revertir el encoding en el notebook de modelo** (menos limpio)

La opción 1 es más correcta. Tendríamos que hablar de cómo tocar el dataClean antes de implementar esto.
