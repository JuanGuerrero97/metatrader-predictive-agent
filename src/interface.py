"""Streamlit interface for the MetaTrader predictive agent.

This module provides a graphical user interface where investors can upload
historical or live market data, run the predictive model, and view
recommendations. It also displays charts and includes risk disclosures.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objs as go

from src.data_utils import load_data, compute_technical_indicators, prepare_features
from src.utils import load_model

DEFAULT_MODEL_PATH = Path("models/model.joblib")


def load_user_data(uploaded_file) -> pd.DataFrame | None:
    """Load CSV data uploaded by the user."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.lower() for c in df.columns]
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    return None


def plot_price(df: pd.DataFrame) -> None:
    """Display an interactive price chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["time"] if "time" in df.columns else df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Precio"
        )
    )
    fig.update_layout(title="Serie de precios", xaxis_title="Tiempo", yaxis_title="Precio")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Asistente de trading MetaTrader", layout="wide")
    st.title("\U0001F4C8 Asistente predictivo para MetaTrader")
    st.markdown(
        """
        Este asistente utiliza un modelo de aprendizaje autom\u00e1tico entrenado con datos de MetaTrader para
        sugerir se\u00f1ales de **compra** o **venta**. Su objetivo es apoyar la toma de decisiones de inversi\u00f3n,
        fomentando pr\u00e1cticas **responsables y sostenibles**. Recuerda que estas se\u00f1ales son probabil\u00edsticas y
        no constituyen asesor\u00eda financiera.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuraci\u00f3n")
    model_path_input = st.sidebar.text_input(
        "Ruta del modelo", value=str(DEFAULT_MODEL_PATH), help="Especifica d\u00f3nde se encuentra el modelo .joblib"
    )
    threshold = st.sidebar.slider(
        "Umbral para se\u00f1al de compra", min_value=0.0, max_value=1.0, value=0.55, step=0.01,
        help="Probabilidad m\u00ednima de compra para recomendar una posici\u00f3n larga"
    )

    uploaded_file = st.file_uploader(
        "Carga un archivo CSV exportado de MetaTrader con columnas como 'time', 'open', 'high', 'low', 'close', 'volume'",
        type=["csv"]
    )

    if uploaded_file is not None:
        user_df = load_user_data(uploaded_file)
        if user_df is not None:
            st.subheader("Datos de entrada")
            st.write(user_df.head())
            plot_price(user_df)

            # Prepare features
            try:
                df_feat = compute_technical_indicators(user_df)
                X, _ = prepare_features(df_feat)
            except Exception as e:
                st.error(f"Error al preparar los datos: {e}")
                return

            # Load model
            model_path = Path(model_path_input)
            if not model_path.exists():
                st.error(f"El modelo no existe en {model_path}. Entrena un modelo usando 'python src/model.py --train'.")
                return
            model = load_model(model_path)
            # Predict probabilities
            proba = model.predict_proba(X)[:, 1]
            predictions = (proba > threshold).astype(int)

            # Show results
            st.subheader("Resultados")
            result_df = pd.DataFrame({
                "probabilidad_compra": proba,
                "se\u00f1al_compra(1)/venta(0)": predictions
            }, index=X.index)
            st.write(result_df.tail())

            # Summary decision
            last_prediction = predictions[-1]
            prob_buy = proba[-1]
            if last_prediction == 1:
                st.success(f"La \u00faltima se\u00f1al sugiere **COMPRAR** con una probabilidad de {prob_buy:.2f}.")
            else:
                st.warning(f"La \u00faltima se\u00f1al sugiere **VENDER/MANTENERSE** con una probabilidad de compra de {prob_buy:.2f}.")

            st.markdown(
                """
                ### Descargo de responsabilidad
                Este sistema no garantiza beneficios y debe utilizarse como un apoyo adicional dentro de una estrategia
                de inversi\u00f3n diversificada. Antes de tomar decisiones, considera factores fundamentales, t\u00e9cnicos y
                de sostenibilidad, as\u00ed como tu perfil de riesgo.
                """
            )


if __name__ == "__main__":
    main()
