# Taller grupal 2 — Métodos Aplicados de Analítica

## Predicción de churn en telecomunicaciones

Proyecto académico para predecir deserción (`resultado = 1`) en clientes pospago. La métrica de evaluación es **AUC-ROC**. La entrega Kaggle requiere un CSV con `id` y `resultado`, donde `resultado` es una probabilidad de churn.

El mejor candidato actual es un **KNN con 61 vecinos, ponderación por distancia y spline de mora de 3 nudos**. Alcanzó AUC público de Kaggle de **0.8199**.

## Arquitectura

Los datos no se unen entre sí:

```text
traintelco.xlsx ─┐
                 ├─ 01_exploracion_preparacion.ipynb ─→ train_preparado.csv
testelco.xlsx ───┘                                      test_preparado.csv

samplesubtelco.csv ─→ solo valida el formato del submission

train_preparado + test_preparado + src/transformaciones.py
                 └─→ notebooks de modelado ─→ submissions y registros
```

- `train` contiene predictores y `resultado`.
- `test` contiene los mismos predictores, pero no `resultado`; por eso recibe las predicciones finales.
- `samplesubtelco.csv` no se usa para entrenar: solo confirma el formato que Kaggle espera.

## Estructura del proyecto

```text
├── CODEX.md                       # Bitácora, resultados y decisiones
├── README.md                      # Guía de reproducción
├── pyproject.toml                 # Dependencias del entorno
├── src/
│   └── transformaciones.py        # Configuraciones y transformaciones reutilizables
├── data/
│   ├── raw/                       # Excel y CSV originales (no se versionan)
│   └── processed/                 # Train/test preparados (no se versionan)
├── notebooks/
│   ├── 01_exploracion_preparacion.ipynb
│   ├── 02_eda_descriptivo.ipynb
│   ├── 03_modelado_logistica_knn.ipynb
│   ├── 04_knn_k57_explicado.ipynb
│   ├── 05_screening_transformaciones_unitarias.ipynb
│   ├── 06_analisis_atipicos_iqr.ipynb
│   └── 07_modelo_final_documentado.ipynb
└── outputs/                       # Métricas, registros y submissions (no se versionan)
```

## Datos requeridos

Copiar sin modificar en `data/raw/`:

- `traintelco.xlsx`
- `testelco.xlsx`
- `samplesubtelco.csv`

Los datos y submissions están excluidos del repositorio para no publicar información del concurso ni archivos generados.

## Instalación

El proyecto usa Python 3.11+ y `uv`.

```powershell
uv sync
uv run jupyter lab
```

Ejecutar Jupyter desde la raíz del proyecto para que funcionen las rutas relativas.

## Orden recomendado de ejecución

### Reproducir el modelo final

1. `01_exploracion_preparacion.ipynb`: genera `data/processed/train_preparado.csv` y `test_preparado.csv`.
2. `07_modelo_final_documentado.ipynb`: valida localmente, entrena con todo train y genera el submission final.

### Entender y desarrollar el modelo

1. `01_exploracion_preparacion.ipynb`: validación estructural y variables derivadas.
2. `02_eda_descriptivo.ipynb`: análisis descriptivo, distribuciones y correlaciones.
3. `05_screening_transformaciones_unitarias.ipynb`: compara transformaciones una por una mediante CV.
4. `03_modelado_logistica_knn.ipynb`: evaluación formal con CV repetida, validación, odds ratios y registro de experimentos.
5. `06_analisis_atipicos_iqr.ipynb`: diagnóstico de extremos con IQR.

`04_knn_k57_explicado.ipynb` es una explicación pedagógica del funcionamiento de KNN. Usa el KNN histórico de 57 vecinos y no representa el modelo final.

## Transformaciones reutilizables

El archivo `src/transformaciones.py` concentra las configuraciones. El notebook 03 las recarga con `importlib.reload`, por lo que basta cambiar:

```python
CONFIGURACION_ACTIVA = "nombre_de_configuracion"
```

Configuraciones relevantes:

- `baseline`: variables originales escaladas.
- `spline_mora_3_nudos`: configuración del modelo final.
- `spline_mora`: spline de mora con cinco nudos.
- `spline_mora_3_nudos_iqr_numericas`: experimento de recorte IQR que no mejoró el modelo.

Las transformaciones que aprenden parámetros —imputación, escalamiento, límites IQR y splines— viven dentro de los pipelines de scikit-learn para evitar fuga de información.

## Resultados principales

| Configuración | Modelo | AUC validación | AUC Kaggle público |
|---|---|---:|---:|
| Baseline | KNN, k=57 | 0.8026 | 0.7943 |
| Spline de mora, 3 nudos | KNN, k=61 | 0.8215 | 0.8199 |
| Spline de mora, 5 nudos | Logística | 0.8164 | — |
| Spline + recorte IQR | KNN, k=57 | 0.8174 | — |

Para el historial completo de experimentos y decisiones, consultar `CODEX.md`.

## Submission

El archivo final debe tener exactamente:

```csv
id,resultado
5,0.1834
8,0.7210
```

`resultado` debe ser una probabilidad entre 0 y 1, no una clase final 0/1.
