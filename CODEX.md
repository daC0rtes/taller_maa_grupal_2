# Bitácora — Taller grupal 2: Clasificación de churn

## 1. Propósito del proyecto

Un operador de telecomunicaciones busca predecir qué clientes pospago abandonarán el servicio (*churn*) y definir una intervención de retención rentable. La entrega predictiva corresponde a un concurso de Kaggle; la entrega académica es un informe CRISP-DM de máximo 12 páginas.

La variable objetivo es `resultado`: `0` significa que no hubo deserción y `1` que hubo deserción. La evaluación oficial es AUC bajo la curva ROC en la base de prueba.

## 2. Preguntas que guían el trabajo

1. ¿Qué variables de perfil, facturación, mora, consumo y relación contractual ayudan a discriminar el churn?
2. ¿Qué desempeño alcanzan regresión logística y KNN bajo una misma validación?
3. ¿Qué modelo generaliza mejor según AUC-ROC?
4. ¿Qué umbral de probabilidad permite intervenir de forma costo/efectiva?

## 3. Datos conocidos

- **Entrenamiento:** muestra aleatoria simple de clientes y contiene `resultado`.
- **Prueba:** muestra adicional no aleatoria, sin `resultado`.
- **Población:** clientes con contratos pospago.
- **Ventana de consumo:** seis meses. Para no churn corresponde al segundo semestre de 2018; en churn son los últimos seis meses antes de la salida, que pudo ocurrir en cualquier punto de ese semestre.

Campos documentados: `id`, `Fecha de nacimiento`, `tipo cliente`, `Factura online`, `Antiguedad equipo`, `Plan de datos`, `Facturación`, `Mora`, `Fecha inicio contrato`, `minutos` y, solo en entrenamiento, `resultado`.

## 4. Decisiones metodológicas iniciales

- La métrica de selección interna será AUC-ROC, alineada con la evaluación oficial.
- La partición de validación será estratificada para preservar la proporción de churn.
- KNN usará escalamiento de variables numéricas; regresión logística también se estimará sobre variables escaladas cuando aplique.
- Las fechas se convertirán a edad y antigüedad de contrato, con validación de valores imposibles y sin usar información posterior a la fecha de corte.
- Se conservará `id` para unir predicciones, pero no será predictor.
- Los archivos originales no se editarán.
- La interpretación de la logística informará odds ratios (`exp(coeficiente)`) y sus límites: son asociaciones condicionadas al modelo, no causalidad.

## 5. Riesgos y controles

| Riesgo | Control propuesto |
|---|---|
| Desbalance de churn | Diagnosticar prevalencia y usar partición estratificada; reportar AUC-ROC. |
| Escalas incompatibles para KNN | Imputación y escalamiento dentro de un `Pipeline`. |
| Fechas inválidas o futuras | Convertir con control de errores y crear indicadores de calidad si procede. |
| Fuga de información | Excluir `id`; revisar variables y transformación antes de validar. |
| Diferencia temporal entre churn/no churn | Documentar la limitación y evitar conclusiones causales sobre consumo. |
| Test no aleatorio | Comparar distribuciones train/test y reportar posible cambio de población. |

## 6. Plan CRISP-DM

| Fase | Evidencia esperada |
|---|---|
| Comprensión del negocio | Objetivo predictivo, uso de retención y supuestos económicos. |
| Comprensión de datos | Diccionario operativo, calidad, univariados y comparación train/test. |
| Preparación | Variables derivadas, imputación, codificación y escalamiento mediante pipelines. |
| Modelado | Regresión logística y KNN con ajuste de hiperparámetros. |
| Evaluación | AUC-ROC, curva ROC, comparación y odds ratios. |
| Despliegue | Submission y política de contacto según beneficio esperado. |

## 7. Bitácora de trabajo

| Fecha | Avance | Decisión / resultado | Pendiente |
|---|---|---|---|
| 2026-11-02 | Inicio del proyecto | Se crea estructura documental y primer notebook, sin ejecución. | Recibir/cargar los archivos de datos y definir entorno. |
| 2026-11-06 | Modelado | Se crea el notebook 03 con validación final, CV repetida, logística, KNN, odds ratios bootstrap y submissions trazables. | Ejecutar, revisar la comparación y elegir el modelo final. |
| 2026-11-07 | Exposición | Se crea un notebook pedagógico independiente para explicar KNN con k=57, sin búsqueda de hiperparámetros. | Ejecutar y usar como apoyo de presentación. |
| 2026-11-07 | Experimentación | Se centralizan transformaciones en `src/transformaciones.py` y se conecta el notebook 03 mediante `importlib.reload`. | Comparar configuraciones y registrar el efecto sobre AUC. |
| 2026-11-07 | Línea base histórica | KNN sin transformaciones avanzadas: k=57, AUC de validación 0,8026 y AUC público de Kaggle 0,7943. | Conservar como referencia frente a nuevos experimentos. |
| 2026-11-07 | Trazabilidad | Se añade registro automático por ejecución en `outputs/experimentos/registro_experimentos.csv`. | Completar AUC público y comentario después de cada submission. |
| 2026-11-07 | Screening unitario | Se crea el notebook 05 para probar log de mora, log de minutos, facturación ajustada y splines individualmente. | Ejecutar screening y promover solo ganadores al notebook 03. |
| 2026-11-07 | Señal spline mora | En evaluación formal reportada por el equipo: logística 0,8164 y KNN 0,8173 de AUC de validación con `spline_mora`. | Ejecutar segunda ronda de sensibilidad de nudos y combinación con minutos. |
| 2026-11-07 | Confirmación spline mora | Corridas formales: 5 nudos favorece logística (AUC 0,8164); 3 nudos favorece KNN en la validación actual (AUC 0,8215, k=61). | Generar submission candidato con KNN 3 nudos y confirmar en Kaggle sin sobreinterpretar la validación repetida. |
| 2026-11-07 | Atípicos IQR | Se diagnosticaron extremos y se configuró recorte IQR de numéricas no spline, preservando mora. | Ejecutar notebook 06 y notebook 03 con `spline_mora_3_nudos_iqr_numericas`. |
| 2026-11-07 | Resultado IQR | KNN con IQR obtuvo AUC 0,8174, inferior a 0,8215 sin IQR; no se promueve esta configuración. | Mantener spline mora 3 nudos sin recorte como mejor candidato KNN. |
| 2026-11-12 | Modelo para compartir | Se crea notebook 07, simplificado y autocontenido, para reproducir el modelo ganador y generar submission. | Compartir en GitHub con datos excluidos del repositorio. |
| 2026-11-04 | Preparación reproducible | Fuentes ubicadas sin modificación en `data/raw/`; se declaró el entorno con `uv` y Python 3.11+ en `pyproject.toml`. El notebook de exploración se ejecutó correctamente con un entorno local compatible: 8.243 filas de train, 2.000 de test, sin faltantes ni IDs duplicados y 16,51 % de churn. | Instalar dependencias en `.venv` con `uv sync` cuando haya acceso a PyPI y continuar con el notebook de modelado. |

## 8. Próximo hito

Revisar los resultados registrados, ejecutar experimentos individuales y escoger el modelo final antes de construir una estrategia de retención.

## 9. Estado de experimentos y aprendizajes

### Implementaciones realizadas

- Se estandarizaron los nombres de columnas y se preservaron las fuentes originales en `data/raw/`.
- Se validaron estructura, faltantes, duplicados e IDs: train tiene 8.243 clientes y test 2.000; no se observaron faltantes ni IDs duplicados.
- Se crearon `edad_anios` y `antiguedad_contrato_meses` desde las fechas, junto con indicadores de valores temporales inválidos.
- Se realizó EDA con distribución de churn, categorías, histogramas, cajas de bigotes, correlaciones, percentiles y comparación train/test.
- Se construyeron pipelines con imputación, codificación one-hot y escalamiento dentro de validación cruzada.
- Se compararon regresión logística y KNN con AUC-ROC, Brier score, curvas ROC, calibración, CV repetida y validación final estratificada.
- Se implementaron odds ratios con intervalos bootstrap, submissions trazables y un notebook pedagógico de KNN con k=57.
- Se centralizaron transformaciones en `src/transformaciones.py`, recargables desde el notebook 03 mediante `importlib.reload`.
- Se añadió un registro automático por ejecución en `outputs/experimentos/registro_experimentos.csv`.

### Experimentos comparables

| Configuración | Modelo | Parámetro seleccionado | AUC CV repetida | AUC validación | Brier validación | AUC público Kaggle | Lectura |
|---|---:|---|---:|---:|---:|---:|---|
| `baseline` | Logística | C=1 | 0,7751 | 0,7791 | 0,1108 | n.d. | Referencia lineal inicial. |
| `baseline` | KNN | k=57, distance | 0,7878 | 0,8026 | 0,1127 | 0,7943 | Mejor resultado observado. |
| `log_mora_minutos_facturacion` | Logística | C=3 | 0,7804 | 0,7836 | 0,1109 | n.d. | Mejora pequeña frente a logística base. |
| `log_mora_minutos_facturacion` | KNN | k=61, distance | 0,7680 | 0,7853 | 0,1149 | 0,7844 | Desempeño inferior al KNN base. |

### Interpretación de la caída

- La transformación conjunta de mora, minutos y facturación redujo KNN: perdió 0,0199 AUC en CV y 0,0173 en validación frente a baseline.
- El mejor k pasó de 57 a 61, cerca del límite superior buscado. Esto indica que la nueva representación empeoró la geometría de distancias y KNN requirió mayor suavizamiento.
- La logística mejoró de 0,7791 a 0,7836 en validación. Las transformaciones no fueron universalmente malas; su efecto depende del modelo.
- El AUC público de Kaggle pertenece al submission de KNN. No reemplaza la validación local, pero la caída simultánea en ambos indicadores es evidencia suficiente para no promover esta configuración final.

### Reglas para próximos experimentos

1. Cambiar una familia de transformaciones por experimento, nunca varias simultáneamente.
2. Mantener baseline como control y conservar el mismo split, CV y grilla de hiperparámetros.
3. Probar `log1p(mora)` y `log1p(minutos)` por separado antes de combinarlas con facturación negativa.
4. Reservar inicialmente splines para regresión logística; para KNN comparar transformaciones monotónicas y escaladas.
5. Completar configuración, métricas, comentario y AUC público en el registro antes de decidir.

### Estado de splines

- Los splines **no se han ejecutado todavía** en un experimento registrado.
- `spline_mora` aplica un spline cúbico de cinco nudos sobre mora dentro del pipeline de regresión logística.
- `spline_minutos` aplica el mismo procedimiento sobre minutos.
- `spline_mora_minutos` prueba ambos splines con cinco nudos; `spline_mora_3_nudos` y `spline_mora_7_nudos` prueban sensibilidad de complejidad.
- La primera señal de `spline_mora` mejoró también KNN, por lo que el notebook 05 compara ambos modelos en esta segunda ronda. La interpretación se mantiene cautelosa: una mejora debe repetirse en CV y luego confirmarse en el notebook 03.

### Segunda ronda de splines: resultados de screening

| Modelo | Configuración | AUC CV media | Desviación CV | Conclusión |
|---|---|---:|---:|---|
| Logística | `spline_mora` (5 nudos) | 0,8108 | 0,0120 | Mejor resultado de logística. |
| Logística | `spline_mora_7_nudos` | 0,8098 | 0,0124 | Prácticamente empatado con 5 nudos; no justifica mayor complejidad. |
| Logística | `spline_mora_minutos` | 0,8098 | 0,0135 | No mejora frente a spline de mora solo. |
| Logística | `spline_mora_3_nudos` | 0,7886 | 0,0110 | Menor flexibilidad pierde señal relevante. |
| KNN | `spline_mora_3_nudos` | 0,8064 | 0,0094 | Mejor resultado puntual de KNN. |
| KNN | `spline_mora_7_nudos` | 0,8064 | 0,0142 | Empata en promedio, pero con mayor variabilidad. |
| KNN | `spline_mora` (5 nudos) | 0,8053 | 0,0130 | Muy cercano; diferencias entre 3, 5 y 7 nudos no son materiales. |
| KNN | `spline_mora_minutos` | 0,7841 | 0,0156 | Empeora frente a spline de mora solo. |

### Lectura de correlaciones

- La correlación de Pearson entre mora y resultado es 0,030, pero el spline de mora mejora claramente el AUC. La señal de mora es no lineal y no se debe evaluar solo con correlación lineal.
- Mora y antigüedad contractual tienen una correlación de -0,712. Esto exige cautela al interpretar sus coeficientes aislados en logística, porque ambas variables comparten información, aunque no invalida su uso predictivo.
- Facturación y minutos se correlacionan 0,398; la relación es moderada y no justifica eliminar automáticamente ninguna variable.
- La matriz de correlación ya cumplió su función. El siguiente gráfico más informativo no es otra matriz, sino la curva estimada de riesgo versus mora del spline ajustado.

### Decisión recomendada

1. Para logística, promover `spline_mora` de 5 nudos al notebook 03; es igual o mejor que 7 nudos y más parsimonioso.
2. Para KNN, promover `spline_mora_3_nudos` al notebook 03 como candidato, pero reconocer que su ventaja sobre 5 y 7 nudos es pequeña frente a la variabilidad CV.
3. No promover `spline_minutos` ni `spline_mora_minutos`, pues no superaron el spline de mora solo.

### Corridas formales de spline mora

| Configuración | Modelo | AUC CV repetida | AUC validación | Brier validación | Parámetro |
|---|---|---:|---:|---:|---|
| `spline_mora` (5 nudos) | Logística | n.d. | 0,8164 | 0,1031 | C=1 |
| `spline_mora` (5 nudos) | KNN | 0,8054 | 0,8133 | 0,1094 | k=59, distance |
| `spline_mora_3_nudos` | Logística | n.d. | 0,7947 | 0,1091 | C=30 |
| `spline_mora_3_nudos` | KNN | 0,8053 | 0,8215 | 0,1095 | k=61, distance |

La diferencia de AUC CV de KNN entre 3 nudos (0,8053) y 5 nudos (0,8054) es de solo 0,0001, mucho menor que la variabilidad entre folds. Por tanto, el 0,8215 de validación con 3 nudos es prometedor, pero debe tratarse como un candidato que necesita confirmación externa, no como una prueba definitiva de superioridad.

### Atípicos IQR: diagnóstico y decisión

| Variable | Atípicos IQR | Tasa churn atípicos | Tasa churn no atípicos | Decisión |
|---|---:|---:|---:|---|
| Facturación | 51 (0,6%) | 29,4% | 16,4% | Conservar filas; recortar solo en experimento de distancias. |
| Mora | 41 (0,5%) | 43,9% | 16,4% | No recortar: contiene señal fuerte de churn y se modela con spline. |
| Minutos | 101 (1,2%) | 26,7% | 16,4% | Conservar filas; recorte IQR experimental antes de escalar. |
| Edad | 79 (1,0%) | 30,4% | 16,4% | Conservar filas; recorte IQR experimental antes de escalar. |
| Antigüedad contrato | 136 (1,7%) | 18,4% | 16,5% | Conservar filas; recorte IQR experimental antes de escalar. |

No se eliminan clientes. La configuración `spline_mora_3_nudos_iqr_numericas` aprende límites IQR en cada fold y limita únicamente las demás variables numéricas; mora se preserva sin recorte para no destruir la señal spline.

La corrida formal IQR confirmó que el recorte no agrega valor: KNN bajó de 0,8215 a 0,8174 en validación y de 0,8053 a 0,8007 en CV repetida; el Brier también pasó de 0,1095 a 0,1103. Por ello, no se enviará esta variante a Kaggle ni se seguirán aplicando recortes IQR por defecto.
