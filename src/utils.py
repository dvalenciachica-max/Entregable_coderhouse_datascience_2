"""Funciones auxiliares del proyecto: carga, limpieza y construcción de atributos."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bakery_sales.csv"
SEED = 42

# Productos de repostería (viennoiserie + pâtisserie) con volumen suficiente para modelar.
VIENNOISERIE = [
    "CROISSANT", "PAIN AU CHOCOLAT", "PAIN AUX RAISINS", "CROISSANT AMANDES",
    "CHAUSSON AUX POMMES", "PAIN CHOCO AMANDES", "BRIOCHE", "KOUIGN AMANN",
]
PATISSERIE = [
    "TARTELETTE", "COOKIE", "ECLAIR", "GRAND FAR BRETON",
    "PARIS BREST", "FINANCIER X5", "FLAN", "MILLES FEUILLES",
]
PRODUCTOS = VIENNOISERIE + PATISSERIE

# Feriados nacionales de Francia en el período del dataset (ene-2021 a sep-2022).
FERIADOS = pd.to_datetime([
    "2021-01-01", "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-13",
    "2021-05-24", "2021-07-14", "2021-08-15", "2021-11-01", "2021-11-11",
    "2021-12-25", "2022-01-01", "2022-04-18", "2022-05-01", "2022-05-08",
    "2022-05-26", "2022-06-06", "2022-07-14", "2022-08-15",
])


def cargar_transacciones(path=DATA_PATH):
    """Lee el CSV original y corrige tipos (fecha, hora y precio con formato '0,90 €')."""
    df = pd.read_csv(path).drop(columns="Unnamed: 0")
    df["date"] = pd.to_datetime(df["date"])
    df["ticket_number"] = df["ticket_number"].astype("int64")
    df["unit_price"] = (
        df["unit_price"].str.extract(r"([\d,]+)")[0].str.replace(",", ".").astype(float)
    )
    return df


def ventas_diarias(trx):
    """Agrega las transacciones a unidades vendidas por día y producto.

    Se usan solo los días en que la panadería abrió (días con al menos un ticket);
    si un producto no se vendió un día abierto, su venta es 0.
    """
    dias_abiertos = trx["date"].drop_duplicates().sort_values()
    rep = trx[trx["article"].isin(PRODUCTOS) & (trx["Quantity"] > 0)]

    unidades = rep.groupby(["date", "article"])["Quantity"].sum()
    precio = rep.groupby(["date", "article"])["unit_price"].median()

    idx = pd.MultiIndex.from_product([dias_abiertos, PRODUCTOS], names=["date", "article"])
    diario = pd.DataFrame({
        "unidades": unidades.reindex(idx, fill_value=0),
        "precio": precio.reindex(idx),
    }).reset_index()

    # Si el producto no se vendió ese día, se toma su último precio conocido.
    diario["precio"] = diario.groupby("article")["precio"].transform(lambda s: s.ffill().bfill())
    return diario


def crear_atributos(diario):
    """Variables de calendario, de categoría y de historia de ventas (sin fuga de información)."""
    df = diario.sort_values(["article", "date"]).copy()

    df["dia_semana"] = df["date"].dt.dayofweek
    df["mes"] = df["date"].dt.month
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["domingo"] = (df["dia_semana"] == 6).astype(int)
    df["feriado"] = df["date"].isin(FERIADOS).astype(int)
    df["temporada_alta"] = df["mes"].isin([7, 8]).astype(int)
    df["categoria"] = np.where(df["article"].isin(VIENNOISERIE), "viennoiserie", "patisserie")

    g = df.groupby("article")["unidades"]
    # Todas las ventanas usan shift(1): solo información disponible antes del día a predecir.
    df["lag_1"] = g.shift(1)
    df["media_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    df["media_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    df["max_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).max())

    # Mismo día de la semana anterior (si ese día estuvo cerrado se usa la media de 7 días).
    semana_ant = df[["date", "article", "unidades"]].copy()
    semana_ant["date"] = semana_ant["date"] + pd.Timedelta(days=7)
    df = df.merge(semana_ant.rename(columns={"unidades": "lag_semana"}),
                  on=["date", "article"], how="left")
    df["lag_semana"] = df["lag_semana"].fillna(df["media_7"])

    return df.dropna(subset=["media_28"]).reset_index(drop=True)
