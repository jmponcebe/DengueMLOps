"""
Dashboard Streamlit para predicción de dengue en Brasil.
Integra API de predicción, visualización geoespacial y monitoreo.
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
import plotly.express as px
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Intentar importar módulos del proyecto
try:
    from src.features.constants import (
        ALL_ENGINEERED_FEATURES, REGION_MAP, REGIONS,
    )
    from src.models.constants import CLASSES
except ImportError:
    ALL_ENGINEERED_FEATURES = []
    CLASSES = [1, 2, 3, 4]
    REGION_MAP = {}
    REGIONS = []

# --- Config ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "historical_api_data"
GEOJSON_PATH = PROJECT_ROOT / "data" / "external" / "brazil_uf_simplified.geojson"
GEOJSON_PATH_FALLBACK = PROJECT_ROOT / "data" / "external" / "brazil_uf.geojson"

ALERT_LABELS = {1: "Verde", 2: "Amarelo", 3: "Laranja", 4: "Vermelho"}
ALERT_COLORS = {1: "#00cc00", 2: "#ffcc00", 3: "#ff6600", 4: "#cc0000"}
ALERT_COLORS_PLOTLY = ["#00cc00", "#ffcc00", "#ff6600", "#cc0000"]
ALERT_DESCRIPTIONS = {
    1: "Situación favorable. Baja incidencia de casos.",
    2: "Atención. Condiciones pueden favorecer transmisión.",
    3: "Alerta. Aumento significativo de incidencia.",
    4: "Emergencia. Alta incidencia, riesgo epidémico.",
}

# UFs y estados
UF_STATES = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
    'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
    'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
    'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul', 'RO': 'Rondônia',
    'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo', 'SE': 'Sergipe',
    'TO': 'Tocantins',
}

# --- Page config ---
st.set_page_config(
    page_title="Dengue Alert Level Prediction - Brazil",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Data helpers (cached) ---
@st.cache_data(ttl=3600)
def load_geojson():
    """Load Brazil UF GeoJSON from local file (simplified for performance)."""
    path = GEOJSON_PATH if GEOJSON_PATH.exists() else GEOJSON_PATH_FALLBACK
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: download from IBGE
    ibge_url = (
        "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
        "?formato=application/vnd.geo+json&intrarregiao=UF"
    )
    resp = requests.get(ibge_url, timeout=15)
    data = resp.json()
    IBGE_TO_UF = {
        '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA',
        '16': 'AP', '17': 'TO', '21': 'MA', '22': 'PI', '23': 'CE',
        '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE',
        '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
        '41': 'PR', '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT',
        '52': 'GO', '53': 'DF',
    }
    for feat in data["features"]:
        cod = feat["properties"]["codarea"]
        feat["properties"]["sigla"] = IBGE_TO_UF.get(cod, cod)
    return data


@st.cache_data(ttl=600)
def load_year_data(year: int) -> pd.DataFrame:
    """Load all UF data for a given year from parquet."""
    if not DATA_DIR.exists():
        return pd.DataFrame()
    try:
        # Read directly from year partitions across all UFs (avoids full scan)
        dfs = []
        cols = [
            "data_iniSE", "SE", "municipio_geocodigo", "municipio_nome",
            "nivel", "pop", "casos_est",
        ]
        for uf_dir in DATA_DIR.iterdir():
            if not uf_dir.is_dir() or not uf_dir.name.startswith("uf="):
                continue
            year_dir = uf_dir / f"year={year}"
            if year_dir.exists():
                part = pd.read_parquet(str(year_dir), columns=cols)
                uf_code = uf_dir.name.split("=")[1]
                part["uf"] = uf_code
                dfs.append(part)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_available_se_range() -> tuple:
    """Get the range of available SE codes across all data."""
    if not DATA_DIR.exists():
        return (2010, 1, 2025, 1)
    try:
        # Find max year from partition dirs (no parquet read needed)
        max_year_found = 2010
        for uf_dir in DATA_DIR.iterdir():
            if not uf_dir.is_dir() or not uf_dir.name.startswith("uf="):
                continue
            for yr_dir in uf_dir.iterdir():
                if yr_dir.is_dir() and yr_dir.name.startswith("year="):
                    yr = int(yr_dir.name.split("=")[1])
                    if yr > max_year_found:
                        max_year_found = yr

        # Read only one UF for the max year to get max SE
        for uf_dir in DATA_DIR.iterdir():
            if not uf_dir.is_dir() or not uf_dir.name.startswith("uf="):
                continue
            yr_path = uf_dir / f"year={max_year_found}"
            if yr_path.exists():
                df = pd.read_parquet(str(yr_path), columns=["SE"])
                if not df.empty:
                    max_se = int(df["SE"].max())
                    return (2010, 1, max_se // 100, max_se % 100)
    except Exception:
        pass
    return (2010, 1, 2025, 1)


def aggregate_by_uf(
    se_data: pd.DataFrame, method: str,
) -> pd.DataFrame:
    """Aggregate municipality-level data to UF level."""
    if method == "mode":
        agg = (
            se_data.groupby("uf", observed=True)["nivel"]
            .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 1)
            .reset_index()
        )
    elif method == "mean":
        agg = (
            se_data.groupby("uf", observed=True)["nivel"]
            .mean().round().astype(int)
            .reset_index()
        )
    else:  # max
        agg = (
            se_data.groupby("uf", observed=True)["nivel"]
            .max().reset_index()
        )

    agg.columns = ["uf", "nivel"]
    agg["label"] = agg["nivel"].map(ALERT_LABELS)
    agg["estado"] = agg["uf"].map(UF_STATES)
    agg["region"] = agg["uf"].map(REGION_MAP)
    return agg


# --- Sidebar ---
def sidebar():
    st.sidebar.title("🦟 Dengue MLOps")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navegación",
        ["Mapa de Alerta", "Predicción", "Información del Modelo", "Monitoreo"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "TFM MLOps — Predicción de Dengue en Brasil\n\n"
        f"API: `{API_URL}`"
    )

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            status = "🟢 Online" if data["model_loaded"] else "🟡 Degradado"
            st.sidebar.success(f"{status} (v{data['version']})")
        else:
            st.sidebar.error("🔴 API no responde")
    except Exception:
        st.sidebar.warning("⚠️ API no disponible")

    return page


# --- Mapa de Alerta ---
def map_page():
    st.title("Mapa de Alerta — Brasil")
    st.markdown(
        "Nivel de alerta de dengue por estado (UF), basado en datos "
        "históricos de InfoDengue/Mosqlimate."
    )

    # Controles
    _, max_year, max_week = get_available_se_range()[1], *get_available_se_range()[2:]

    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1.2])
    with col_c1:
        use_latest = st.checkbox("Datos más recientes", value=True)
    with col_c2:
        year = st.selectbox(
            "Año", range(max_year, 2009, -1), index=0, disabled=use_latest,
        )
    with col_c3:
        default_week = max_week if use_latest else 1
        se_num = st.slider(
            "Semana epidemiológica", 1, 53, default_week, disabled=use_latest,
        )
    with col_c4:
        agg_method = st.selectbox(
            "Agregación por UF",
            ["Nivel predominante (moda)", "Nivel medio", "Nivel máximo"],
        )

    # Determinar SE a mostrar
    if use_latest:
        sel_year, sel_week = max_year, max_week
    else:
        sel_year, sel_week = year, se_num

    se_code = sel_year * 100 + sel_week

    # Cargar datos
    df_year = load_year_data(sel_year)
    if df_year.empty:
        st.warning(f"No hay datos disponibles para el año {sel_year}.")
        return

    se_data = df_year[df_year["SE"] == se_code]
    if se_data.empty:
        # Buscar SE más cercana disponible
        available_ses = sorted(df_year["SE"].unique())
        if available_ses:
            closest = min(available_ses, key=lambda x: abs(x - se_code))
            st.info(
                f"SE {se_code} no disponible. Mostrando SE {closest} "
                f"(la más cercana con datos)."
            )
            se_data = df_year[df_year["SE"] == closest]
            se_code = closest
        else:
            st.warning("No hay datos para este período.")
            return

    # Agregación
    method_key = (
        "mode" if "moda" in agg_method
        else ("mean" if "medio" in agg_method else "max")
    )
    agg = aggregate_by_uf(se_data, method_key)

    # Header con fecha
    se_week = se_code % 100
    se_yr = se_code // 100
    date_str = se_data["data_iniSE"].iloc[0] if "data_iniSE" in se_data.columns else ""
    st.markdown(
        f"**Semana epidemiológica {se_week}/{se_yr}** "
        f"{'(' + str(date_str) + ')' if date_str else ''} — "
        f"{len(se_data)} registros municipales, "
        f"{se_data['uf'].nunique()} estados"
    )

    # --- Mapa choropleth (mapbox-based, more reliable with custom GeoJSON) ---
    geojson = load_geojson()

    # nivel as string for discrete color mapping
    agg["nivel_str"] = agg["nivel"].astype(str)

    fig = px.choropleth_mapbox(
        agg,
        geojson=geojson,
        locations="uf",
        featureidkey="properties.sigla",
        color="nivel_str",
        color_discrete_map={
            "1": ALERT_COLORS[1], "2": ALERT_COLORS[2],
            "3": ALERT_COLORS[3], "4": ALERT_COLORS[4],
        },
        category_orders={"nivel_str": ["1", "2", "3", "4"]},
        hover_name="estado",
        hover_data={
            "nivel": True, "label": True, "region": True,
            "uf": False, "nivel_str": False,
        },
        labels={
            "nivel": "Nivel", "label": "Alerta", "region": "Región",
            "nivel_str": "Nivel",
        },
        mapbox_style="carto-positron",
        center={"lat": -14.2, "lon": -51.9},
        zoom=3,
        opacity=0.85,
    )
    fig.update_traces(
        marker_line_color="#333333",
        marker_line_width=1,
    )
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="Nivel de Alerta",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.01,
            xanchor="center", x=0.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Leyenda y resumen ---
    col_leg, col_stats = st.columns([1, 1])
    with col_leg:
        st.markdown("**Escala de alerta InfoDengue:**")
        for nivel, label in ALERT_LABELS.items():
            color = ALERT_COLORS[nivel]
            n_ufs = len(agg[agg["nivel"] == nivel])
            st.markdown(
                f'<span style="color:{color}; font-size:1.3em;">●</span> '
                f"**Nivel {nivel} — {label}**: {n_ufs} estados",
                unsafe_allow_html=True,
            )

    with col_stats:
        st.markdown("**Resumen nacional:**")
        nivel_dist = agg["nivel"].value_counts().sort_index()
        fig_pie = px.pie(
            names=[f"Nivel {n} ({ALERT_LABELS[n]})" for n in nivel_dist.index],
            values=nivel_dist.values,
            color_discrete_sequence=[ALERT_COLORS[n] for n in nivel_dist.index],
        )
        fig_pie.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Detalle por municipio ---
    st.markdown("---")
    st.subheader("Detalle por municipio")

    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        ufs_available = sorted(se_data["uf"].unique())
        selected_uf = st.selectbox(
            "Estado",
            ufs_available,
            format_func=lambda x: f"{x} — {UF_STATES.get(x, x)}",
        )

    muni = (
        se_data[se_data["uf"] == selected_uf]
        [["municipio_nome", "nivel", "pop", "casos_est"]]
        .copy()
    )
    muni.columns = ["Municipio", "Nivel", "Población", "Casos est."]
    muni = muni.sort_values("Nivel", ascending=False).reset_index(drop=True)

    with col_info:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Municipios", len(muni))
        with m2:
            st.metric("Nivel máximo", int(muni["Nivel"].max()))
        with m3:
            mode_val = muni["Nivel"].mode()
            st.metric(
                "Nivel predominante",
                int(mode_val.iloc[0]) if len(mode_val) > 0 else "-",
            )
        with m4:
            pop_total = muni["Población"].sum()
            st.metric(
                "Población total",
                f"{pop_total:,.0f}" if pd.notna(pop_total) else "-",
            )

    # Distribución de niveles en el estado
    col_chart, col_table = st.columns([1, 2])
    with col_chart:
        muni_dist = muni["Nivel"].value_counts().sort_index()
        fig_bar = px.bar(
            x=[f"N{n}" for n in muni_dist.index],
            y=muni_dist.values,
            color=[ALERT_LABELS.get(n, str(n)) for n in muni_dist.index],
            color_discrete_map={
                v: ALERT_COLORS[k] for k, v in ALERT_LABELS.items()
            },
            labels={"x": "Nivel", "y": "Municipios"},
        )
        fig_bar.update_layout(
            height=280, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        # Colorear filas según nivel
        def color_nivel(val):
            colors = {
                1: "background-color: #00cc0030",
                2: "background-color: #ffcc0030",
                3: "background-color: #ff660030",
                4: "background-color: #cc000030",
            }
            return colors.get(val, "")

        styled = muni.style.map(color_nivel, subset=["Nivel"])
        st.dataframe(styled, use_container_width=True, height=300)


# --- Predicción ---
def prediction_page():
    st.title("Predicción de Nivel de Alerta")
    st.markdown(
        "Ingresa los datos del municipio y las condiciones climáticas "
        "para obtener una predicción del nivel de alerta de dengue."
    )

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Datos de entrada")

        # Temporal
        st.markdown("**Información temporal**")
        col_m, col_se = st.columns(2)
        with col_m:
            month = st.slider("Mes", 1, 12, value=3)
        with col_se:
            # SE range coherent with selected month
            se_min = max(1, int((month - 1) * (52 / 12)) + 1)
            se_max = min(52, int(month * (52 / 12)))
            se_default = (se_min + se_max) // 2
            se = st.slider(
                "Semana epidemiológica", se_min, se_max,
                value=se_default,
                help=f"Mes {month} → SE {se_min}–{se_max}",
            )

        # Geográfica
        st.markdown("**Información geográfica**")
        col_uf, col_pop = st.columns(2)
        with col_uf:
            uf = st.selectbox(
                "Estado (UF)",
                options=list(UF_STATES.keys()),
                format_func=lambda x: f"{x} - {UF_STATES[x]}",
                index=list(UF_STATES.keys()).index("RJ"),
            )
        with col_pop:
            population = st.number_input(
                "Población del municipio",
                min_value=1000, max_value=15_000_000,
                value=500_000, step=10_000,
            )

        # Climática
        st.markdown("**Datos climáticos** (con lag de 4-8 semanas)")
        col_t, col_h = st.columns(2)
        with col_t:
            temp_lag8w = st.number_input(
                "Temperatura media (lag 8 sem.)", 10.0, 40.0, 26.0, 0.5,
            )
            temp_roll12w = st.number_input(
                "Media móvil temperatura (12 sem.)", 10.0, 40.0, 25.5, 0.5,
            )
        with col_h:
            humid_roll4w = st.number_input(
                "Media móvil humedad (4 sem.)", 30.0, 100.0, 78.0, 1.0,
            )
            temp_x_humid = temp_lag8w * humid_roll4w  # auto-calculado

        predict_btn = st.button("🔍 Predecir", type="primary", use_container_width=True)

    # Construir features
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    se_sin = float(np.sin(2 * np.pi * se / 52))
    se_cos = float(np.cos(2 * np.pi * se / 52))
    is_peak = 1 if month in [1, 2, 3, 4] else 0
    quarter = ((month - 1) // 3) + 1
    pop_log = float(np.log1p(population))

    # Region one-hot
    region = REGION_MAP.get(uf, "Sudeste")
    region_ne = 1 if region == "Nordeste" else 0
    region_co = 1 if region == "Centro-Oeste" else 0
    region_se = 1 if region == "Sudeste" else 0
    region_s = 1 if region == "Sul" else 0

    if predict_btn:
        payload = {
            "month_sin": month_sin, "month_cos": month_cos,
            "se_sin": se_sin, "se_cos": se_cos,
            "is_peak_season": is_peak, "quarter": quarter,
            "tempmed_lag8w": temp_lag8w, "tempmed_roll12w": temp_roll12w,
            "umidmed_roll4w": humid_roll4w, "temp_x_humid_lag4w": temp_x_humid,
            "pop_log": pop_log,
            "region_Nordeste": region_ne, "region_Centro-Oeste": region_co,
            "region_Sudeste": region_se, "region_Sul": region_s,
        }

        with col_result:
            st.subheader("Resultado")
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["prediction"]
                    nivel = pred["nivel"]
                    label = pred["label"]
                    color = pred["color"]
                    proba = pred["probabilities"]

                    # Resultado principal
                    st.markdown(
                        f"""
                        <div style="background-color: {color}20; border-left: 6px solid {color};
                             padding: 20px; border-radius: 8px; margin-bottom: 16px;">
                            <h2 style="color: {color}; margin: 0;">Nivel {nivel} — {label}</h2>
                            <p style="margin: 8px 0 0 0;">{ALERT_DESCRIPTIONS.get(nivel, '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Probabilidades
                    st.markdown("**Distribución de probabilidades**")
                    prob_df = pd.DataFrame(
                        {"Nivel": list(proba.keys()), "Probabilidad": list(proba.values())}
                    )
                    colors = list(ALERT_COLORS.values())
                    fig = px.bar(
                        prob_df, x="Nivel", y="Probabilidad",
                        color="Nivel",
                        color_discrete_sequence=colors,
                    )
                    fig.update_layout(
                        height=300, showlegend=False,
                        yaxis_range=[0, 1], yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detalles
                    with st.expander("Detalles de la predicción"):
                        st.json(data)

                else:
                    st.error(f"Error de la API: {resp.status_code} — {resp.text}")
            except requests.ConnectionError:
                st.error(
                    "No se pudo conectar con la API. "
                    "Asegúrate de que está corriendo en `localhost:8000`."
                )
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        with col_result:
            st.subheader("Resultado")
            st.info("Configura los parámetros y pulsa **Predecir**.")

            # Escala de referencia
            st.markdown("**Escala de alerta InfoDengue:**")
            for nivel, label in ALERT_LABELS.items():
                color = ALERT_COLORS[nivel]
                st.markdown(
                    f'<span style="color:{color}; font-weight:bold;">●</span> '
                    f'**Nivel {nivel} — {label}**: {ALERT_DESCRIPTIONS[nivel]}',
                    unsafe_allow_html=True,
                )


# --- Model Info ---
def model_info_page():
    st.title("Información del Modelo")

    try:
        resp = requests.get(f"{API_URL}/model/info", timeout=5)
        if resp.status_code != 200:
            st.error("No se pudo obtener información del modelo")
            return
        info = resp.json()
    except Exception:
        st.warning("API no disponible. Mostrando información estática.")
        info = {
            "name": "dengue-alertlevel-classifier",
            "version": "1", "alias": "champion",
            "algorithm": "XGBClassifier",
            "n_features": 15,
            "features": ALL_ENGINEERED_FEATURES if ALL_ENGINEERED_FEATURES else [],
            "target": "nivel",
            "classes": {str(k): v for k, v in ALERT_LABELS.items()},
            "metrics": {"macro_f1": 0.39, "accuracy": 0.88, "cohen_kappa": 0.33},
            "training_period": "2010-2021",
            "description": "Modelo champion XGBoost con balanced weights.",
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algoritmo", info["algorithm"])
    with col2:
        st.metric("Macro F1 (test)", f"{info['metrics'].get('macro_f1', 0):.2f}")
    with col3:
        st.metric("Cohen's Kappa", f"{info['metrics'].get('cohen_kappa', 0):.2f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Detalles del modelo")
        st.markdown(f"""
        | Propiedad | Valor |
        |---|---|
        | **Nombre** | `{info['name']}` |
        | **Versión** | {info['version']} |
        | **Alias** | {info['alias']} |
        | **Período entrenamiento** | {info['training_period']} |
        | **N° features** | {info['n_features']} |
        | **Target** | `{info['target']}` (niveles 1-4) |
        """)

        st.markdown(f"**Descripción:** {info['description']}")

    with col_b:
        st.subheader("Features de producción")
        features = info.get("features", [])
        if features:
            categories = {
                "Temporales": [f for f in features if any(
                    t in f for t in ["month_", "se_", "peak", "quarter"]
                )],
                "Climáticas": [f for f in features if any(
                    t in f for t in ["temp", "umid", "humid"]
                )],
                "Geográficas": [f for f in features if any(
                    t in f for t in ["pop_", "region_"]
                )],
            }
            for cat, feats in categories.items():
                with st.expander(f"{cat} ({len(feats)})", expanded=True):
                    for f in feats:
                        st.code(f, language=None)

    # Métricas detalladas
    st.markdown("---")
    st.subheader("Métricas del Champion")

    metrics = info.get("metrics", {})
    if metrics:
        metrics_df = pd.DataFrame(
            {"Métrica": list(metrics.keys()), "Valor": list(metrics.values())}
        )
        fig = px.bar(
            metrics_df, x="Métrica", y="Valor",
            color="Valor", color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=350, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Artefactos
    st.subheader("Artefactos del modelo")
    artifacts_dir = PROJECT_ROOT / "mlflow-artifacts" / "6386d99c6830446080b7e29945d05af9" / "artifacts"
    if artifacts_dir.exists():
        col_cm, col_fi = st.columns(2)
        cm_path = artifacts_dir / "confusion_matrix.png"
        fi_path = artifacts_dir / "feature_importance.png"
        if cm_path.exists():
            with col_cm:
                st.image(str(cm_path), caption="Matriz de Confusión (Test 2024)")
        if fi_path.exists():
            with col_fi:
                st.image(str(fi_path), caption="Feature Importance")

        # SHAP
        shap_path = artifacts_dir / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP Summary Plot", width=600)
    else:
        st.info("Artefactos no encontrados localmente.")


# --- Monitoreo ---
def monitoring_page():
    st.title("Monitoreo del Modelo")

    # Predicciones logueadas
    log_path = PROJECT_ROOT / "monitoring" / "predictions_log.csv"
    drift_report_path = PROJECT_ROOT / "monitoring" / "reports"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicciones recientes")
        if log_path.exists():
            df = pd.read_csv(log_path)
            st.metric("Total predicciones", len(df))

            # Distribución
            if "predicted_nivel" in df.columns:
                dist = df["predicted_nivel"].value_counts().sort_index()
                fig = px.pie(
                    names=[ALERT_LABELS.get(n, str(n)) for n in dist.index],
                    values=dist.values,
                    color_discrete_sequence=list(ALERT_COLORS.values()),
                    title="Distribución de niveles predichos",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Últimas predicciones
            with st.expander("Últimas 20 predicciones"):
                st.dataframe(df.tail(20), use_container_width=True)
        else:
            st.info("Aún no hay predicciones registradas.")

    with col2:
        st.subheader("Data Drift")

        # Buscar reportes Evidently
        if drift_report_path.exists():
            reports = sorted(drift_report_path.glob("*.html"), reverse=True)
            if reports:
                selected = st.selectbox(
                    "Seleccionar reporte",
                    reports,
                    format_func=lambda p: p.stem,
                )
                if selected:
                    with open(selected, "r", encoding="utf-8") as f:
                        st_components.html(f.read(), height=600, scrolling=True)
            else:
                st.info("No hay reportes de drift disponibles.")
        else:
            st.info(
                "Los reportes de monitoreo se generarán automáticamente.\n\n"
                "Ejecuta `python -m src.monitoring.drift_detector` para generar un reporte."
            )

    # Opción para generar reporte
    st.markdown("---")
    if st.button("📊 Generar reporte de drift", type="secondary"):
        try:
            requests.post(f"{API_URL}/monitoring/flush", timeout=5)
            st.success("Buffer de predicciones flushed. Ejecuta el drift detector para generar el reporte.")
        except Exception:
            st.warning("API no disponible para flush.")


# --- Main ---
def main():
    page = sidebar()

    if page == "Mapa de Alerta":
        map_page()
    elif page == "Predicción":
        prediction_page()
    elif page == "Información del Modelo":
        model_info_page()
    elif page == "Monitoreo":
        monitoring_page()


if __name__ == "__main__":
    main()
