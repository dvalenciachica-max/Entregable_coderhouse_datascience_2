# Predicción de demanda diaria de repostería — Panadería francesa

Proyecto final del curso de Data Science (Coderhouse).

## Problema

Una panadería hornea cada mañana productos de repostería perecederos, como croissants, pains au chocolat, tartaletas y éclairs. Si hornea de más, genera merma. Si hornea de menos, pierde ventas. El objetivo es **predecir cuántas unidades de cada producto se van a vender cada día**, para planificar la producción.

| | |
|---|---|
| Tipo de tarea | Regresión supervisada |
| Variable objetivo | `unidades`: unidades vendidas por producto y día |
| Métrica principal | RMSE (unidades). Métricas de apoyo: MAE y R² |
| Baseline del negocio | Hornear lo mismo que el mismo día de la semana anterior |

## Dataset

[French Bakery Daily Sales](https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales) (Kaggle): 234.005 líneas de tickets de una panadería francesa entre el 02-01-2021 y el 30-09-2022. La copia usada está en `data/bakery_sales.csv`.

El análisis se centra en 16 productos de repostería: 8 de viennoiserie y 8 de pâtisserie.

## Estructura del repositorio

```
├── notebook_final.ipynb   # Proceso completo: problema, EDA, atributos, modelos, optimización, evaluación, SHAP y conclusiones
├── src/utils.py           # Carga, limpieza y construcción de atributos
├── data/bakery_sales.csv  # Dataset original
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
git clone <https://github.com/dvalenciachica-max/Entregable_coderhouse_datascience_2.git> 
cd <carpeta_del_repositorio>
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebook_final.ipynb
```

Ejecutá **Kernel → Restart & Run All**. Todo el proceso corre en unos 2 minutos. La semilla está fija (`SEED = 42`), así que los resultados son reproducibles.

## Metodología

1. **EDA:** calidad de los datos (devoluciones, precios guardados como texto, duplicados, cierres semanales), outliers con el criterio IQR y patrones semanales, estacionales y de precios.
2. **Ingeniería de atributos:** variables de calendario (día de la semana, domingo, temporada alta, feriados de Francia), categoría de producto, precio y variables de historia sin fuga de información (venta del día anterior, del mismo día de la semana anterior y medias móviles de 7 y 28 días). One-Hot Encoding y escalado dentro de un `Pipeline`.
3. **Modelos:** Ridge, Árbol de decisión, Random Forest y XGBoost.
4. **Optimización:** GridSearchCV y RandomizedSearchCV con validación cruzada temporal (`TimeSeriesSplit`, 5 folds).
5. **Evaluación:** test temporal de julio a septiembre de 2022, con el modelo elegido por el RMSE de validación cruzada.
6. **Explicabilidad:** SHAP (`TreeExplainer`), con importancia global y explicación de una predicción individual.

## Resultados

| Modelo | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Baseline (mismo día semana anterior) | — | 13,00 | 5,95 | 0,769 |
| Ridge (optimizado) | 8,29 | 10,96 | 5,07 | 0,836 |
| Árbol de decisión (optimizado) | 7,53 | 9,88 | 4,61 | 0,867 |
| XGBoost (optimizado) | 7,14 | 9,42 | 4,22 | 0,879 |
| **Random Forest (optimizado)** | **6,95** | **9,16** | **4,33** | **0,885** |

- El modelo elegido, **Random Forest**, reduce el RMSE un **30 %** respecto de la regla actual del negocio, y baja alrededor de un **25 %** el error de producción valorizado en euros.
- **SHAP:** las variables más influyentes son la venta del mismo día de la semana anterior y las medias móviles, seguidas por el día de la semana y la temporada alta.
- **Limitaciones:** el período de test cubre solo la temporada alta, y en los productos de muy baja venta la mejora sobre la regla simple es pequeña.
