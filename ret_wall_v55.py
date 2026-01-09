# app_streamlit_clean.py
# Streamlit UI with a clean schematic and correct bearing pressure visualization.
# Uses locked computation engine from engine_locked.py (derived from ret_wall_v50.py).

import math
import numpy as np
import pandas as pd
import streamlit as st
# --- Display formatting ---
pd.options.display.float_format = "{:.3f}".format

def _round_df_3(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all float columns rounded to 3 decimals."""
    if df is None:
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]) or pd.api.types.is_numeric_dtype(out[c]):
            # round only floats; keep ints intact
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(3)
    return out

def st_scrollable_table(df: pd.DataFrame, *, height_px: int = 320):
    """Render a horizontally-scrollable table (mobile-friendly) without Streamlit dataframe toolbars.

    Styling is handled via CSS (centered cells, colored header, zebra striping).
    """
    if df is None:
        st.info("No table to display.")
        return

    df2 = _round_df_3(df)

    html = df2.to_html(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        classes="rw-table",
        border=0,
    )

    st.markdown(
        f'''
        <div class="rw-table-wrap" style="max-height:{height_px}px;">
          <div class="rw-table-inner">
            {html}
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="Your App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Hide per-element toolbar icons (download/search/fullscreen) */
div[data-testid="stElementToolbar"] {display: none !important;}
div.modebar {display: none !important;}

/* ===== Custom tables: centered + colored header + zebra + mobile horizontal scroll ===== */
.rw-table-wrap{
  width:100%;
  overflow-x:auto;
  overflow-y:auto;
  border:1px solid rgba(49,51,63,0.20);
  border-radius:10px;
}
.rw-table-inner{
  min-width:max-content;
  padding:6px 8px;
}
table.rw-table{
  border-collapse:collapse;
  width:100%;
  font-size:14px;
}
table.rw-table thead th{
  background:#2f6fa5;
  color:#ffffff;
  text-align:center !important;
  padding:8px 10px;
  border:1px solid rgba(49,51,63,0.18);
  white-space:nowrap;
  font-weight:600;
}
table.rw-table tbody td{
  text-align:center !important;
  vertical-align:middle !important;
  padding:6px 10px;
  border:1px solid rgba(49,51,63,0.12);
  white-space:nowrap;
}
table.rw-table tbody tr:nth-child(even){background:#f3f6f9;}
table.rw-table tbody tr:nth-child(odd){background:#ffffff;}


/* --- عمومی --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* دکمه Deploy/Cloud در برخی نسخه‌ها */
.stDeployButton {display: none !important;}
div[data-testid="stDeployButton"] {display: none !important;}

/* --- فقط موبایل: آیکون‌های بالا و پایین --- */
@media (max-width: 768px) {

  /* آیکون‌های بالای صفحه (سمت راست) - بدون اینکه دکمه سایدبار را از بین ببرد */
  div[data-testid="stToolbarActions"] {display: none !important;}
  div[data-testid="stHeaderActionElements"] {display: none !important;}

  /* آیکون‌های پایین ( + و 👑 و … ) */
  div[data-testid="stStatusWidget"] {display: none !important;}
  div[data-testid="stDecoration"] {display: none !important;}

  button[title="Create new app"],
  button[title="Streamlit Cloud"],
  button[aria-label="Create new app"],
  button[aria-label="Streamlit Cloud"],
  button[aria-label="Open app menu"] {
    display: none !important;
  }

  /* نسخه‌های موبایل گاهی این‌ها را به صورت fixed می‌گذارند */
  div[style*="position: fixed"][style*="bottom"] {display: none !important;}
  div[style*="position:fixed"][style*="bottom"] {display: none !important;}
}
</style>
""", unsafe_allow_html=True)
import streamlit.components.v1 as components

components.html("""
<script>
(function () {

  // فقط روی موبایل
  if (!window.matchMedia("(max-width: 768px)").matches) return;

  function removeFABs() {
    document.querySelectorAll('button, div').forEach(el => {
      const style = window.getComputedStyle(el);

      // فقط المان‌های شناور
      if (style.position !== 'fixed') return;

      const rect = el.getBoundingClientRect();

      // گوشه پایین صفحه
      const nearBottom = rect.top > window.innerHeight * 0.6;
      const nearRight  = rect.left > window.innerWidth * 0.5;

      // اندازه دکمه‌های FAB
      const sizeOK = rect.width >= 40 && rect.width <= 90 &&
                     rect.height >= 40 && rect.height <= 90;

      // آیکون SVG ( + یا 👑 )
      const hasSVG = el.querySelector && el.querySelector('svg');

      if (nearBottom && nearRight && sizeOK && hasSVG) {
        el.style.display = 'none';
      }
    });
  }

  // چندبار اجرا چون Streamlit بعداً دوباره inject می‌کند
  let tries = 0;
  const interval = setInterval(() => {
    removeFABs();
    tries++;
    if (tries > 25) clearInterval(interval);
  }, 400);

})();
</script>
""", height=0)

# ---- ادامه کد اپ ----

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Patch

# --- Schematic colors ---
CONCRETE_COLOR = "#B0B0B0"  # concrete gray
SOIL_BACK_COLOR = "#E6D3A3"  # soil (backfill)
SOIL_FRONT_COLOR = "#E6D3A3" # soil (front cover)

from engine_locked import compute, rad

def _clamp(v, a, b):
    return max(a, min(b, v))

def draw_schematic_clean(res, mode="simple", show_pressures=True, show_dims=True):
    """
    Clean schematic:
      - Clear layering (wall, soil, pressures, bearing)
      - Dimensions moved outside geometry to avoid overlap
      - Bearing pressure: trapezoid if full contact, triangle if uplift (per engine b_contact)
      - b_contact reported and shown
    """
    B = res["B"]; t_f = res["t_f"]; h = res["h_stem"]
    t_top = res["t_top"]; t_bot = res["t_bot"]
    L_toe = res["L_toe"]; L_heel = res["L_heel"]
    beta = res["beta"]
    H = res["H_active"]
    H_front = res["H_front"]
    t_cover_front = res["t_cover_front"]

    # Bearing-related from engine
    V = res["R_total"]
    e_base = res["e_base"]
    kern = res["kern"]
    b_contact = float(res["b_contact"])

    # Derived for display (avoid sign confusion)
    q_avg = V / B if abs(B) > 1e-12 else float("nan")
    q_toe = q_avg * (1 + 6*e_base/B) if abs(B) > 1e-12 else float("nan")
    q_heel= q_avg * (1 - 6*e_base/B) if abs(B) > 1e-12 else float("nan")
    # For uplift case per engine logic (e_base >= B/6): max at toe, zero at heel
    q_max_uplift = (2*V/b_contact) if (b_contact and b_contact > 1e-9) else float("nan")

    x_R = float(res["x_R"])

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # -------------------------
    # Geometry
    # -------------------------
    # Footing
    footing = np.array([[0,0],[B,0],[B,t_f],[0,t_f]])
    # Filled concrete footing + outline
    ax.add_patch(Polygon(footing, closed=True, fill=True, facecolor=CONCRETE_COLOR, edgecolor="none", zorder=4))
    ax.add_patch(Polygon(footing, closed=True, fill=False, linewidth=1, edgecolor="black", zorder=5))

    # Stem: front face vertical at x=L_toe; back face sloped
    x_front = L_toe
    stem = np.array([
        [x_front, t_f],
        [x_front, t_f+h],
        [x_front+t_top, t_f+h],
        [x_front+t_bot, t_f],
    ])
    # Filled concrete stem + outline
    ax.add_patch(Polygon(stem, closed=True, fill=True, facecolor=CONCRETE_COLOR, edgecolor="none", zorder=5))
    ax.add_patch(Polygon(stem, closed=True, fill=False, linewidth=1, edgecolor="black", zorder=6))

    # Back face endpoints (for soil intersection)
    xb0, yb0 = x_front + t_bot, t_f
    xb1, yb1 = x_front + t_top, t_f + h
    dx, dy = xb1 - xb0, yb1 - yb0

    # Backfill surface line: through (B, H) with slope tan(beta) (rising to the right)
    tb = math.tan(rad(beta))

    # Intersect surface line with back face (segment)
    denom = dy - tb*dx
    if abs(denom) < 1e-12:
        t_int = 1.0
    else:
        # solve yb0+dy*t = H + tb*(x - B), x=xb0+dx*t
        t_int = (H + tb*(xb0 - B) - yb0)/denom
    t_int = _clamp(t_int, 0.0, 1.0)
    x_int = xb0 + dx*t_int
    y_int = yb0 + dy*t_int

    # Soil polygon (behind wall) - no gap
    soil_poly = np.array([
        [xb0, t_f],
        [B, t_f],
        [B, H],
        [x_int, y_int],
    ])
    ax.add_patch(Polygon(soil_poly, closed=True, facecolor=SOIL_BACK_COLOR, edgecolor="none", alpha=0.70, zorder=1))

    # Front soil cover on toe (if any)
    if t_cover_front > 1e-9:
        toe_soil = np.array([[0,t_f],[L_toe,t_f],[L_toe,t_f+t_cover_front],[0,t_f+t_cover_front]])
        ax.add_patch(Polygon(toe_soil, closed=True, facecolor=SOIL_FRONT_COLOR, edgecolor="none", alpha=0.70, zorder=1))

    # -------------------------
    # Pressure shapes (contrast)
    # -------------------------
    if show_pressures:
        # Active pressure triangle drawn on x=B line from y=t_f to y=H
        xP = B
        pa = np.array([[xP, 0],[xP, H],[xP+0.14*B, 0]])
        ax.add_patch(Polygon(pa, closed=True, facecolor="#F5B041", edgecolor="#B9770E", alpha=0.85, zorder=2))
        ax.text(xP+0.16*B, (t_f+H)/2, "Pa", fontsize=10, va="center")

        # Passive pressure at toe, y=0..H_front
        pp = np.array([[0,0],[0,H_front],[-0.12*B,0]])
        ax.add_patch(Polygon(pp, closed=True, facecolor="#58D68D", edgecolor="#1D8348", alpha=0.85, zorder=2))
        ax.text(-0.14*B, H_front*0.5, "Pp", fontsize=10, va="center", ha="right")

    # Track how far below the footing the bearing diagram extends (for axis limits)
    bearing_depth = 0.0

    # -------------------------
    # Bearing pressure diagram (ALWAYS drawn **below** the footing)
    # Convention: compression is positive, drawn downward.
    if show_pressures:
        # Engine may return negative bearing on one side (uplift). For plotting, do NOT draw negative pressures.
        q_toe_draw = max(float(q_toe), 0.0)
        q_heel_draw = max(float(q_heel), 0.0)

        # Determine whether uplift occurs (any side loses contact)
        uplift = (q_toe < 0.0) or (q_heel < 0.0) or (b_contact < B - 1e-9)

        # Vertical scaling for pressure shape
        q_max_draw = max(q_toe_draw, q_heel_draw, 1e-9)
        scale = 0.45 * B / q_max_draw  # plotting height scale
        bearing_depth = q_max_draw * scale

        y0 = 0.0  # footing bottom (ground line in this schematic)
        if uplift:
            # Contact-only triangle over b' contact length.
            # Decide uplift side from e_base sign (+ toward toe):
            #  - e_base >= 0 => resultant toward toe => uplift at heel => contact from toe: x in [0, b_contact], peak at toe
            #  - e_base < 0  => resultant toward heel => uplift at toe => contact near heel: x in [B-b_contact, B], peak at heel
            qmax = max(q_toe_draw, q_heel_draw, 0.0)  # peak at compressed edge
            h = qmax * scale

            if e_base >= 0:
                x_left, x_right = 0.0, b_contact
                x_peak = x_left
            else:
                x_left, x_right = B - b_contact, B
                x_peak = x_right

            # triangle drawn downward
            poly_q = [(x_left, y0), (x_right, y0), (x_peak, y0 - h)]
            ax.add_patch(Polygon(poly_q, closed=True, facecolor="#A855F7", edgecolor="#6D28D9", alpha=0.25, linewidth=2))
            ax.plot([x_left, x_right], [y0, y0], color="#6D28D9", linestyle="--", linewidth=1)

            # label b_contact
            ax.text((x_left + x_right) / 2, y0 - (0.80 + 0.15*max(t_f,1.0)), f"b_contact = {b_contact:.3f} m", ha="center", va="top", fontsize=8, color="#4C1D95")
            ax.text((x_left + x_right) / 2, y0 - (1.10 + 0.20*max(t_f,1.0)), "Bearing (triangle)", ha="center", va="top", fontsize=8, color="#4C1D95")
        else:
            # Full contact trapezoid
            poly_q = [(0, y0), (B, y0), (B, y0 - q_heel_draw * scale), (0, y0 - q_toe_draw * scale)]
            ax.add_patch(Polygon(poly_q, closed=True, facecolor="#A855F7", edgecolor="#6D28D9", alpha=0.20, linewidth=2))
            ax.text(B/2, y0 - (0.80 + 0.15*max(t_f,1.0)), f"b_contact = {B:.3f} m", ha="center", va="top", fontsize=8, color="#4C1D95")
            ax.text(B/2, y0 - (1.10 + 0.20*max(t_f,1.0)), "Bearing (trapezoid)", ha="center", va="top", fontsize=8, color="#4C1D95")
# Resultant location
    # -------------------------
    ax.annotate("", xy=(x_R, t_f+0.4), xytext=(x_R, t_f + 1.40),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"), zorder=21)
    ax.text(x_R, t_f + 1.45, "R_total", fontsize=8, ha="center")

    # -------------------------
    # Dimensions outside geometry
    # -------------------------
    if show_dims:
        # Base width B
        ydim = -0.50*max(1.0,t_f)
        ax.annotate("", xy=(0, ydim), xytext=(B, ydim),
                    arrowprops=dict(arrowstyle="<->", lw=1.2, color="0.30"))
        ax.text(B/2, ydim+0.1*max(1.0,t_f), f"B={B:.3f} m", fontsize=8, ha="center", color="0.25")

        # Height H
        xdim = B + 0.25*B
        ax.annotate("", xy=(xdim, 0.0), xytext=(xdim, H),
                    arrowprops=dict(arrowstyle="<->", lw=1.2, color="0.30"))
        ax.text(xdim+0.02*B, (0.0+H)/2, f"H={H:.3f} m", fontsize=8, rotation=90, va="center", color="0.25")

        # L_toe and L_heel labels (minimal)
        ax.text(L_toe/2, t_f+0.1, f"Toe {L_toe:.3f}", fontsize=8, ha="center", color="0.20")
        ax.text(B - L_heel/2, t_f+0.1, f"Heel {L_heel:.3f}", fontsize=8, ha="center", color="0.20")

    # Extra info (detailed)
    if mode == "detailed":
        ax.text(0.02*B, H+0.25, f"Ka={res['Ka']:.4f}  Kp={res['Kp_full']:.3f}  Kp_reduced={res['Kp_reduced']:.3f}", fontsize=10)
        ax.text(0.02*B, H+0.06, f"x_R={x_R:.3f} m  e_base={e_base:.3f} m  (B/6={kern:.3f})", fontsize=10)
        ax.text(0.02*B, H-0.13, f"σmax={res['sigma_max']:.1f} kPa  σmin={res['sigma_min']:.1f} kPa  b'={b_contact:.3f} m", fontsize=10)


    # -------------------------
    # Legend
    # -------------------------
    legend_items = [
        Patch(facecolor=CONCRETE_COLOR, edgecolor="black", label="Concrete"),
        Patch(facecolor=SOIL_BACK_COLOR, edgecolor="none", alpha=0.70, label="Soil"),
    ]
    if show_pressures:
        legend_items += [
            Patch(facecolor="#F5B041", edgecolor="#B9770E", alpha=0.85, label="Pa"),
            Patch(facecolor="#58D68D", edgecolor="#1D8348", alpha=0.85, label="Pp"),
            Patch(facecolor="#A855F7", edgecolor="#6D28D9", alpha=0.20, label="Bearing"),
        ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    # view
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.35*B, 1.35*B)
    # Dynamic y-limits to avoid clipping (bearing diagram + labels)
    y_pad = max(0.8, 0.10*max(B, H))
    y_min = min(-bearing_depth - 0.6, -0.35*B - 0.2*t_f, -1.5*t_f)
    y_max = max(H + 0.9, 0.35*B + y_pad)
    ax.set_ylim(y_min, y_max)
    ax.margins(x=0.03, y=0.03)
    ax.axis("off")
    fig.tight_layout(pad=1.0)
    return fig

def main():
    st.set_page_config(page_title="Cantilever Retaining Wall Calculator", layout="wide")

    st.markdown("""
<style>
/* Responsive title */
.rw-title-box {
    background-color: #2f6fa5;
    padding: 14px 20px;
    border-radius: 12px;
    margin-bottom: 18px;
}

.rw-title-box h1 {
    color: white;
    margin: 0;
    font-weight: 600;
    text-align: center;
    font-size: 32px;   /* Desktop */
}

/* Mobile */
@media (max-width: 768px) {
    .rw-title-box {
        padding: 10px 14px;
    }
    .rw-title-box h1 {
        font-size: 24px;  /* Mobile */
    }
}
</style>

<div class="rw-title-box">
    <h1>Cantilever Retaining Wall Calculator</h1>
</div>
""", unsafe_allow_html=True)

    # --- 👈 Mobile sidebar hint (اینجا کپی کن) ---
            
    with st.sidebar:
        st.header("Inputs")
        h_stem = st.number_input("stem height (m)", value=7.0, min_value=0.1, step=0.1)
        t_top  = st.number_input("stem thickness at top (m)", value=0.4, min_value=0.05, step=0.05)
        t_bot  = st.number_input("stem thickness at bottom (m)", value=0.7, min_value=0.05, step=0.05)
        t_f    = st.number_input("footing thickness (m)", value=0.7, min_value=0.05, step=0.05)
        L_toe  = st.number_input("toe length (m)", value=1.5, min_value=0.05, step=0.1)
        L_heel = st.number_input("heel length (m)", value=3.2, min_value=0.05, step=0.1)

        st.divider()
        gamma_backfill = st.number_input("unit weight of backfill, [γf] (kN/m³)", value=20.0, min_value=1.0, step=0.5)
        phi = st.number_input("internal friction angle of backfill, [φ] (deg)", value=30.0, min_value=0.0, max_value=60.0, step=0.5)
        delta = st.number_input("wall-backfill interface friction angel [δ] (deg)", value=10.0, min_value=0.0, max_value=45.0, step=0.5)
        beta = st.number_input("slope of backfill, [β] (deg)", value=15.0, min_value=0.0, max_value=45.0, step=0.5)

        st.divider()
        gamma_concrete = st.number_input("unit weight of concrete [γc] (kN/m³)", value=24.0, min_value=10.0, step=0.5)
        t_cover_front = st.number_input("depth of front soil over the toe (m)", value=0.6, min_value=0.0, step=0.05)
        passive_reduction = st.number_input("Passive reduction factor", value=0.5, min_value=0.0, max_value=1.0, step=0.05)

        st.divider()
        mu = st.number_input("μ (base friction)", value=0.58, min_value=0.0, max_value=1.5, step=0.01)
        q_allow = st.number_input("q_allow (kPa)", value=200.0, min_value=1.0, step=10.0)
        st.divider()
        # Schematic settings (fixed)
        mode = "simple"
        show_pressures = True
        show_dims = True

    inputs = {
        "h_stem": h_stem, "t_top": t_top, "t_bot": t_bot, "t_f": t_f,
        "L_toe": L_toe, "L_heel": L_heel,
        "gamma_backfill": gamma_backfill, "phi": phi, "delta": delta, "beta": beta,
        "gamma_concrete": gamma_concrete,
        "t_cover_front": t_cover_front,
        "mu": mu, "q_allow": q_allow,
        "passive_reduction": passive_reduction,
    }

    res = compute(inputs)

    # Top metrics
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("B (m)", f"{res['B']:.3f}")
    r2.metric("H_active (m)", f"{res['H_active']:.3f}")
    r3.metric("Ka", f"{res['Ka']:.6f}")
    r4.metric("Kp (full)", f"{res['Kp_full']:.3f}")

    r5,r6,r7,r8 = st.columns(4)
    r5.metric("Pa_H (kN/m)", f"{res['P_H']:.3f}")
    r6.metric("Pa_V (kN/m)", f"{res['P_V']:.3f}")
    r7.metric("Pp (kN/m)", f"{res['Pp']:.3f}")
    r8.metric("R_total (kN/m)", f"{res['R_total']:.3f}")

    st.divider()

    left, right = st.columns([1.25, 1.0])

    with left:
        st.subheader("Wall Schematic")
        fig = draw_schematic_clean(res, mode=mode, show_pressures=show_pressures, show_dims=show_dims)
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Stability Checks")
        c1,c2 = st.columns(2)
        c1.metric("FS sliding", f"{res['FS_sl']:.3f}", "OK ✅" if res["FS_sl"] >= 1.5 else "NOT OK ❌")
        c2.metric("FS overturning", f"{res['FS_ot']:.3f}", "OK ✅" if res["FS_ot"] >= 2.0 else "NOT OK ❌")

        st.divider()
        b1,b2 = st.columns(2)
        b1.metric("f_max (kPa)", f"{res['sigma_max']:.1f}", "OK ✅" if res["sigma_max"] <= res["q_allow"] else "NOT OK ❌")
        b2.metric("f_min (kPa)", f"{res['sigma_min']:.1f}", "OK ✅" if res["sigma_min"] >= 0 else "NOT OK ❌")

        st.caption(f"b_contact = {res['b_contact']:.3f} m  |  B/6 = {res['kern']:.3f} m")
        st.divider()
        d1,d2 = st.columns(2)
        d1.metric("x_R (m from toe)", f"{res['x_R']:.3f}")
        d2.metric("e_base (m)", f"{res['e_base']:.3f}", "OK ✅" if res["e_base"] <= res["kern"] else "NOT OK ❌")

    st.divider()
    st.subheader("Vertical loads")
    st_scrollable_table(res["df_vertical"], height_px=260)
            
    with st.expander("Outputs"):
        keys = [
        "B","H_active","alpha","i","Ka","Kp_full","Kp_reduced","H_front",
        "P_H","P_V","Pp","R_total","M_stab","x_R","e_load",
        "M_ot","M_net","e_base","kern","b_contact",
        "sigma_max","sigma_min","FS_ot","FS_sl"
        ]

        UNITS = {
        "B": "m",
        "H_active": "m",
        "H_front": "m",
        "alpha": "deg",
        "i": "deg",
        "Ka": "-",
        "Kp_full": "-",
        "Kp_reduced": "-",
        "P_H": "kN/m",
        "P_V": "kN/m",
        "Pp": "kN/m",
        "R_total": "kN/m",
        "M_stab": "kN·m/m",
        "M_ot": "kN·m/m",
        "M_net": "kN·m/m",
        "x_R": "m",
        "e_load": "m",
        "e_base": "m",
        "kern": "m",
        "b_contact": "m",
        "sigma_max": "kPa",
        "sigma_min": "kPa",
        "FS_ot": "-",
        "FS_sl": "-"
        }

    # چیزی که در ستون Symbol نمایش داده می‌شود
        SYMBOL = {
        "sigma_max": "f_max",
        "sigma_min": "f_min",
        "FS_ot": "FS_overturning",
        "FS_sl": "FS_sliding",
        "alpha": "α",
        "M_stab": "M_stabilizing",
        "M_ot": "M_overturning",
        "P_H": "Pa_H",
        "P_V": "Pa_V",
        }

    # توضیح خوانا برای کاربر
        DESC = {
        "B": "Footing width",
        "H_active": "Height of active pressure",
        "H_front": "Front soil height",
        "alpha": "Backface batter angle from horizontal",
        "i": "Backface batter angle from vertical",
        "Ka": "Active earth pressure coefficient",
        "Kp_full": "Passive coefficient (full)",
        "Kp_reduced": "Passive coefficient (reduced)",
        "P_H": "Resultant horizontal earth force",
        "P_V": "Resultant vertical earth force",
        "Pp": "Passive resultant force",
        "R_total": "Total vertical load",
        "M_stab": "Stabilizing moment",
        "M_ot": "Overturning moment",
        "M_net": "Net bending moment at the base",
        "x_R": "Resultant location from toe",
        "e_load": "Eccentricity of loads",
        "e_base": "Base eccentricity",
        "kern": "Middle third (kernel) limit",
        "b_contact": "Contact length",
        "sigma_max": "Maximum bearing pressure",
        "sigma_min": "Minimum bearing pressure",
        "FS_ot": "Factor of safety against overturning",
        "FS_sl": "Factor of safety against sliding",
        }

        dbg = pd.DataFrame({
        "Symbol": [SYMBOL.get(k, k) for k in keys],
        "Description": [DESC.get(k, "") for k in keys],
        "Unit": [UNITS.get(k, "") for k in keys],
        "Value": [res.get(k) for k in keys],
        })

        st_scrollable_table(dbg, height_px=360)


if __name__ == "__main__":
    main()
