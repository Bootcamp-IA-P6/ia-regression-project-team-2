# Transformación Logarítmica del Target

**Fecha:** 2026-02-19

## Qué es y por qué ayuda

### El problema: la distribución sesgada

Imagina que tienes estos precios en tu dataset:

```
150.000€, 200.000€, 180.000€, 250.000€, 160.000€, 23.000.000€
```

La regresión lineal intenta minimizar los errores de todas las filas. Pero ese piso de 23M€ tiene errores enormes comparado con los demás. Para minimizar ese error gigante, el modelo "distorsiona" sus coeficientes, empeorando las predicciones para los pisos normales (que son el 99% del dataset).

Esto es exactamente lo que pasa con tu RMSE (643K€) vs MAE (355K€): hay unas pocas predicciones con errores enormes que inflan el RMSE.

### La solución: trabajar en escala logarítmica

El logaritmo "comprime" los valores grandes y "expande" los pequeños. Mira cómo cambian las distancias:

```
Precio real:      150K  →  200K  →  500K  →  1M   →  5M   →  23M
Diferencia real:       50K     300K     500K   4M      18M
                                                              ↑ enorme

log(precio):      11.9  →  12.2  →  13.1  →  13.8  →  15.4  →  16.9
Diferencia log:       0.3     0.9     0.7    1.6      1.5
                                                              ↑ manejable
```

En escala logarítmica, los pisos de lujo ya no "gritan" tanto. El modelo puede aprender patrones del mercado general sin que los outliers lo distorsionen.

### La intuición económica

Los precios de inmuebles en realidad funcionan en porcentajes, no en euros absolutos:

- "Un piso en Salamanca vale un 30% más que en Chamberí"
- "Un piso exterior vale un 20% más que uno interior"

Cuando trabajas con log(precio), el modelo aprende directamente estas relaciones porcentuales. Eso es más natural para el problema.

---

## Cómo implementarlo

### El proceso en 3 pasos

```
1. Transformar:   y_log = log(precio)
2. Entrenar:      modelo.fit(X_train, y_log_train)
3. Revertir:      precio_predicho = exp(y_log_predicho)
```

### Por qué `log1p` y no `log`

`np.log1p(x)` calcula `log(x + 1)`. El +1 protege contra `log(0)`, que daría -infinito. Con precios nunca tendrás un 0, pero es buena práctica.

Para revertir se usa `np.expm1(x)`, que calcula `exp(x) - 1`.

---

## Código completo para el notebook de modelo

Estos son los cambios respecto a lo que tienes ahora. Solo cambia la definición de `y` y la evaluación:

```python
# --- CAMBIO 1: Transformar el target ---
# En vez de:
y = df['PrecioActual']

# Usar:
y = np.log1p(df['PrecioActual'])
# Ahora y contiene log(precio), no el precio directamente

# El split y el encoding se hacen igual que antes, no cambia nada.

# --- CAMBIO 2: Entrenar igual que antes ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)  # y_train ahora son valores log

ridge_final = Ridge(alpha=mejor_alpha)
ridge_final.fit(X_train_scaled, y_train)

# --- CAMBIO 3: Al predecir, revertir el log ---
# El modelo predice log(precio), hay que convertirlo a euros reales
y_pred_lr_log    = lr.predict(X_test_scaled)
y_pred_lr_euros  = np.expm1(y_pred_lr_log)  # ← convertir de vuelta a euros

y_pred_ridge_log   = ridge_final.predict(X_test_scaled)
y_pred_ridge_euros = np.expm1(y_pred_ridge_log)

# --- CAMBIO 4: Evaluar con precios reales (euros) ---
# y_test está en log, hay que convertirlo también para comparar en euros
y_test_euros = np.expm1(y_test)

print("=== Regresión Lineal ===")
print(f"R²:   {r2_score(y_test, y_pred_lr_log):.4f}")
# El R² se puede calcular en escala log (más estable) o en euros (más intuitivo)
# En escala log es más representativo de cómo aprendió el modelo

print(f"MAE (euros):  {mean_absolute_error(y_test_euros, y_pred_lr_euros):,.0f} €")
print(f"RMSE (euros): {np.sqrt(mean_squared_error(y_test_euros, y_pred_lr_euros)):,.0f} €")

# Métrica adicional útil: error porcentual medio
# Cuánto se equivoca el modelo en porcentaje respecto al precio real
mape = np.mean(np.abs((y_test_euros - y_pred_lr_euros) / y_test_euros)) * 100
print(f"MAPE: {mape:.1f}%  (error porcentual medio)")
```

---

## Cómo interpretar los resultados después del cambio

### El R² no cambia mucho

El R² mide la proporción de varianza explicada. Sigue siendo válido, pero ahora mide la varianza del log(precio), no del precio. Un R² de 0.85 en escala log es muy bueno.

### El MAE y RMSE bajarán drásticamente en euros

Antes, los pisos de lujo generaban errores de millones que inflaban estas métricas. Ahora el modelo ya no los persigue tanto.

### La nueva métrica: MAPE

El **MAPE** (Mean Absolute Percentage Error) es la más intuitiva después del log transform:

```
MAPE = 15% significa que el modelo se equivoca de media un 15% del precio real
```

Es mucho más fácil de interpretar que "MAE = 355.000€". Con precios que van de 150K a 23M€, un error absoluto no tiene el mismo significado para todos los inmuebles.

---

## Qué resultado esperar

Con el encoding mejorado + log transform, un R² de 0.80-0.85 sería razonable para regresión lineal con este dataset. Si se queda por debajo de 0.80, probablemente hay algo más a investigar (outliers extremos, features que faltan, etc.).
