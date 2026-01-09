import streamlit as st
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retaining Wall Calculator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# GLOBAL CSS (TABLES)
# =========================
st.markdown("""
<style>

/* Hide Streamlit toolbars completely */
div[data-testid="stElementToolbar"],
div[data-testid="stToolbarActions"],
div[data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* Table container for horizontal scroll */
.table-container {
    overflow-x: auto;
    width: 100%;
    margin-top: 8px;
}

/* Base table */
.table-container table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
}

/* Header styling */
.table-container thead th {
    background-color: #2f6fa5;
    color: white;
    text-align: center !important;
    padding: 8px;
    border: 1px solid #cfcfcf;
    white-space: nowrap;
}

/* Body cells */
.table-container tbody td {
    text-align: center !important;
    vertical-align: middle !important;
    padding: 6px 8px;
    border: 1px solid #dddddd;
    white-space: nowrap;
}

/* Zebra striping */
.table-container tbody tr:nth-child(even) {
    background-color: #f3f6f9;
}

.table-container tbody tr:nth-child(odd) {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("Cantilever Retaining Wall Calculator")
st.caption("Simple mode – Desktop & Mobile friendly")

# =========================
# INPUTS (SIDEBAR – SIMPLE ONLY)
# =========================
with st.sidebar:
    st.header("Inputs")

    H = st.number_input("Wall height H (m)", value=6.0, step=0.1)
    gamma = st.number_input("Soil unit weight γ (kN/m³)", value=18.0, step=0.1)
    phi = st.number_input("Soil friction angle φ (deg)", value=30.0, step=1.0)
    mu = st.number_input("Base friction coefficient μ", value=0.5, step=0.05)

# =========================
# CALCULATIONS (SIMPLE)
# =========================
Ka = np.tan(np.radians(45 - phi / 2)) ** 2
Pa = 0.5 * gamma * H**2 * Ka
W = gamma * H * 1.0
FS_sliding = (mu * W) / Pa if Pa != 0 else np.nan

# =========================
# RESULTS TABLE
# =========================
df_results = pd.DataFrame({
    "Check": ["Active Earth Pressure", "Wall Weight", "Sliding FS"],
    "Value": [Pa, W, FS_sliding],
    "Unit": ["kN/m", "kN/m", "-"]
}).round(3)

st.subheader("Design Results")
st.markdown(
    f"""
    <div class="table-container">
        {df_results.to_html(index=False)}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# INTERMEDIATE VALUES
# =========================
df_debug = pd.DataFrame({
    "Parameter": ["Ka", "γ", "φ"],
    "Value": [Ka, gamma, phi],
    "Unit": ["-", "kN/m³", "deg"]
}).round(3)

st.subheader("Intermediate Values")
st.markdown(
    f"""
    <div class="table-container">
        {df_debug.to_html(index=False)}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# FOOTER
# =========================
st.caption("Download, fullscreen and advanced table tools are intentionally disabled for a clean mobile experience.")
