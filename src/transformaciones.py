"""Transformaciones experimentales para los modelos de churn.

Cambiar CONFIGURACION_ACTIVA permite repetir el notebook 03 sin editar su
lógica de validación. Las variables derivadas aquí se calculan fila a fila,
por lo que no usan información de validación ni de test. Imputación,
escalamiento y splines se crean dentro del Pipeline de scikit-learn.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


# Cambiar solo este nombre para ejecutar otro experimento desde el notebook 03.
CONFIGURACION_ACTIVA = "spline_mora_3_nudos_iqr_numericas"

CONFIGURACIONES = {
    "baseline": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": [],
    },
    "solo_log_mora": {
        "log_mora": True,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": [],
    },
    "solo_log_minutos": {
        "log_mora": False,
        "log_minutos": True,
        "ajustar_facturacion": False,
        "splines": [],
    },
    "solo_facturacion_ajustada": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": True,
        "splines": [],
    },
    "spline_mora": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["mora"],
        "n_knots": 5,
    },
    "spline_minutos": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["minutos"],
        "n_knots": 5,
    },
    "spline_mora_minutos": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["mora", "minutos"],
        "n_knots": 5,
    },
    "spline_mora_3_nudos": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["mora"],
        "n_knots": 3,
    },
    "spline_mora_3_nudos_iqr_numericas": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["mora"],
        "n_knots": 3,
        "recortar_iqr_numericas": True,
        "recortar_iqr_splines": False,
    },
    "spline_mora_7_nudos": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": False,
        "splines": ["mora"],
        "n_knots": 7,
    },
    "log_mora_minutos_facturacion": {
        "log_mora": True,
        "log_minutos": True,
        "ajustar_facturacion": True,
        "splines": [],
    },
    "splines_mora_minutos": {
        "log_mora": False,
        "log_minutos": False,
        "ajustar_facturacion": True,
        "splines": ["mora", "minutos"],
    },
}

BASE_NUMERICAS = [
    "antiguedad_equipo",
    "facturacion",
    "mora",
    "minutos",
    "edad_anios",
    "antiguedad_contrato_meses",
]
BASE_CATEGORICAS = [
    "tipo_cliente",
    "factura_online",
    "plan_datos",
    "edad_invalida",
    "antiguedad_contrato_invalida",
]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Limita extremos según límites IQR aprendidos únicamente del entrenamiento."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        datos = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(datos, 25, axis=0)
        q3 = np.nanpercentile(datos, 75, axis=0)
        iqr = q3 - q1
        self.limite_inferior_ = q1 - self.factor * iqr
        self.limite_superior_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inferior_, self.limite_superior_)

    def get_feature_names_out(self, input_features=None):
        """Conserva los nombres: el recorte no crea ni elimina columnas."""
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


def obtener_configuracion(nombre: str | None = None) -> dict:
    """Devuelve una copia de una configuración para evitar cambios accidentales."""
    nombre = nombre or CONFIGURACION_ACTIVA
    if nombre not in CONFIGURACIONES:
        raise ValueError(f"Configuración desconocida: {nombre}. Opciones: {list(CONFIGURACIONES)}")
    return deepcopy(CONFIGURACIONES[nombre])


def aplicar_transformaciones(df: pd.DataFrame, configuracion: dict) -> pd.DataFrame:
    """Crea variables determinísticas y conserva una base sin modificar."""
    salida = df.copy()

    if configuracion["log_mora"]:
        salida["mora_tiene_valor"] = (salida["mora"] > 0).astype("int8")
        salida["mora_log1p"] = np.log1p(salida["mora"].clip(lower=0))
        salida = salida.drop(columns="mora")

    if configuracion["log_minutos"]:
        salida["minutos_log1p"] = np.log1p(salida["minutos"].clip(lower=0))
        salida = salida.drop(columns="minutos")

    if configuracion["ajustar_facturacion"]:
        # No se elimina información: se separan cobro positivo, saldo negativo y su magnitud.
        salida["facturacion_negativa"] = (salida["facturacion"] < 0).astype("int8")
        salida["facturacion_positiva"] = salida["facturacion"].clip(lower=0)
        salida["magnitud_facturacion_negativa"] = (-salida["facturacion"].clip(upper=0))
        salida = salida.drop(columns="facturacion")

    return salida


def obtener_grupos(configuracion: dict) -> tuple[list[str], list[str], list[str]]:
    """Devuelve numéricas lineales, numéricas con spline y categóricas."""
    numericas = BASE_NUMERICAS.copy()
    categoricas = BASE_CATEGORICAS.copy()

    if configuracion["log_mora"]:
        numericas.remove("mora")
        numericas.append("mora_log1p")
        categoricas.append("mora_tiene_valor")

    if configuracion["log_minutos"]:
        numericas.remove("minutos")
        numericas.append("minutos_log1p")

    if configuracion["ajustar_facturacion"]:
        numericas.remove("facturacion")
        numericas.extend(["facturacion_positiva", "magnitud_facturacion_negativa"])
        categoricas.append("facturacion_negativa")

    splines = configuracion["splines"]
    numericas_lineales = [col for col in numericas if col not in splines]
    return numericas_lineales, splines, categoricas


def crear_preprocesador(configuracion: dict) -> ColumnTransformer:
    """Crea el preprocesador que será ajustado dentro de cada fold."""
    numericas, splines, categoricas = obtener_grupos(configuracion)

    pasos_numericas = [("imputar", SimpleImputer(strategy="median"))]
    if configuracion.get("recortar_iqr_numericas", False):
        pasos_numericas.append(("recortar_iqr", IQRClipper()))
    pasos_numericas.append(("escalar", StandardScaler()))

    transformadores = [
        (
            "numericas",
            Pipeline(pasos_numericas),
            numericas,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
            ]),
            categoricas,
        ),
    ]

    if splines:
        pasos_splines = [("imputar", SimpleImputer(strategy="median"))]
        if configuracion.get("recortar_iqr_splines", False):
            pasos_splines.append(("recortar_iqr", IQRClipper()))
        pasos_splines.extend([
            ("spline", SplineTransformer(n_knots=configuracion.get("n_knots", 5), degree=3, knots="quantile", extrapolation="linear")),
            ("escalar", StandardScaler()),
        ])
        transformadores.append(
            (
                "splines",
                Pipeline(pasos_splines),
                splines,
            )
        )

    return ColumnTransformer(transformadores, remainder="drop")
