# engine_locked.py
# Locked computation engine extracted verbatim from ret_wall_v50.py (no formula changes)

import math
import numpy as np
import pandas as pd

def deg(x): 
    return x * 180.0 / math.pi

def rad(x): 
    return x * math.pi / 180.0

def safe_div(a, b, default=float("nan")):
    return a / b if abs(b) > 1e-12 else default

def coulomb_Ka_excel(alpha_deg, phi_deg, delta_deg, beta_deg, i_deg):
    """Exact Ka per Excel cells Q10..W11:
       Ka = (cos(delta + i) * sin(alpha + phi)^2) / (sin(alpha)^2 * sin(alpha - delta) * (1 + sqrt((sin(delta+phi)*sin(phi-beta)) / (sin(alpha-delta)*sin(alpha+beta))))^2 )
    """
    alpha = rad(alpha_deg)
    phi   = rad(phi_deg)
    delta = rad(delta_deg)
    beta  = rad(beta_deg)
    i     = rad(i_deg)

    Q10 = math.cos(delta + i)
    R10 = (math.sin(alpha + phi))**2
    Q11 = (math.sin(alpha))**2
    R11 = math.sin(alpha - delta)

    T11 = math.sin(delta + phi)
    U11 = math.sin(phi - beta)
    T12 = math.sin(alpha - delta)
    U12 = math.sin(alpha + beta)

    inside = safe_div(T11 * U11, T12 * U12, default=float("nan"))
    if inside < 0:
        # keep consistent but avoid crash
        inside = float("nan")
    W11 = (1.0 + math.sqrt(inside))**2

    Ka = safe_div(Q10 * R10, Q11 * R11 * W11, default=float("nan"))
    return Ka

def rankine_Kp(phi_deg):
    return (math.tan(rad(45.0 + 0.5*phi_deg)))**2

# -----------------------------
# Core calculations (mirrors Sheet1)
# -----------------------------
def compute(inputs):
    h_stem = inputs["h_stem"]
    t_top  = inputs["t_top"]
    t_bot  = inputs["t_bot"]
    t_f    = inputs["t_f"]
    L_toe  = inputs["L_toe"]
    L_heel = inputs["L_heel"]
    gamma_soil = inputs["gamma_backfill"]
    phi    = inputs["phi"]
    delta  = inputs["delta"]
    beta   = inputs["beta"]
    gamma_c = inputs["gamma_concrete"]
    t_cover_front = inputs["t_cover_front"]
    mu     = inputs["mu"]
    q_allow= inputs["q_allow"]
    passive_reduction = inputs["passive_reduction"]

    # B (m) = L_heel + t_bot + L_toe  (as Excel B10 = B9 + B6 + B8)
    B = L_heel + t_bot + L_toe

    # H_active (m) = t_f + h_stem + (L_heel + t_bot - t_top)*tan(beta)  (Excel F4)
    H_active = t_f + h_stem + (L_heel + t_bot - t_top) * math.tan(rad(beta))

    # alpha, i
    # Per your request: use alpha = 90 deg and i = 0 deg
    alpha = math.degrees(math.atan(h_stem/(t_bot-t_top)))
    i_deg = 90.0-alpha

    # Ka (Excel F10)
    Ka = coulomb_Ka_excel(alpha_deg=alpha, phi_deg=phi, delta_deg=delta, beta_deg=beta, i_deg=i_deg)

    # Active forces (Excel F12, F14)
    P_H = 0.5 * Ka * gamma_soil * (H_active**2)
    P_V = P_H * math.tan(rad(delta))

    # Passive (Excel F18,F19,F21); H_front = t_cover_front + t_f
    Kp_full = rankine_Kp(phi)
    Kp_reduced  = passive_reduction * Kp_full
    H_front = t_cover_front + t_f
    Pp = passive_reduction * 0.5 * Kp_full * gamma_soil * (H_front**2)

    # Vertical components table (Excel H4..K13)
    items = []

    # Footing concrete: W = B * t_f * gamma_c ; x = 0.5*B
    W_foot = B * t_f * gamma_c
    x_foot = 0.5 * B
    items.append(("Footing Concrete", W_foot, x_foot))

    # Stem concrete (rectangle): W = t_top * h_stem * gamma_c ; x = L_toe + 0.5*t_top
    W_stem_rect = t_top * h_stem * gamma_c
    x_stem_rect = L_toe + 0.5 * t_top
    items.append(("Stem Concrete (rectangluar part)", W_stem_rect, x_stem_rect))

    # Stem concrete (triangle): W = 0.5*(t_bot - t_top)*h_stem*gamma_c ; x = L_toe + t_top + (1/3)*(t_bot - t_top)
    W_stem_tri = 0.5 * (t_bot - t_top) * h_stem * gamma_c
    x_stem_tri = L_toe + t_top + (1.0/3.0) * (t_bot - t_top)
    items.append(("Stem Concrete (triangular part)", W_stem_tri, x_stem_tri))

    # Soil on heel rectangle: W = L_heel * h_stem * gamma_soil ; x = L_toe + t_bot + 0.5*L_heel
    W_soil_heel_rect = L_heel * h_stem * gamma_soil
    x_soil_heel_rect = L_toe + t_bot + 0.5 * L_heel
    items.append(("Soil on Heel (rectangular part)", W_soil_heel_rect, x_soil_heel_rect))

    # Soil on heel triangle over stem slope: W = 0.5*(t_bot - t_top)*h_stem*gamma_soil ; x = L_toe + t_top + (2/3)*(t_bot - t_top)
    W_soil_heel_tri = 0.5 * (t_bot - t_top) * h_stem * gamma_soil
    x_soil_heel_tri = L_toe + t_top + (2.0/3.0) * (t_bot - t_top)
    items.append(("Soil on Heel (triangular part)", W_soil_heel_tri, x_soil_heel_tri))

    # Soil above heel due to backfill slope: W = 0.5*(H_active - t_f - h_stem)*(L_heel + t_bot - t_top)*gamma_soil
    W_soil_slope = 0.5 * (H_active - t_f - h_stem) * (L_heel + t_bot - t_top) * gamma_soil
    x_soil_slope = B - (1.0/3.0) * (B - L_toe - t_top)
    items.append(("Soil on Heel (slopped part)", W_soil_slope, x_soil_slope))

    # Soil on toe (front cover): W = L_toe * t_cover_front * gamma_soil ; x = 0.5*L_toe
    W_soil_toe = L_toe * t_cover_front * gamma_soil
    x_soil_toe = 0.5 * L_toe
    items.append(("Soil on Toe", W_soil_toe, x_soil_toe))

    # Pv at heel end: W = P_V ; x = B
    items.append(("Pa-V", P_V, B))

    df = pd.DataFrame(items, columns=["Component","W (kN/m)","x from toe (m)"])
    df["M_about_toe (kN·m/m)"] = df["W (kN/m)"] * df["x from toe (m)"]

    R_total = df["W (kN/m)"].sum()             # Excel I15
    M_stab  = df["M_about_toe (kN·m/m)"].sum() # Excel I17

    x_R = safe_div(M_stab, R_total, default=B/2)  # Excel I16
    e_load = x_R - 0.5*B                          # Excel K16 (named e_load)

    # Overturning moment (Excel I19): M_ot = P_H * (H_active/3)
    M_ot = P_H * (H_active/3.0)

    # M_net (Excel I21): = M_ot - R_total*e_load
    M_net = M_ot - R_total * e_load

    # Eccentricity e_base (Excel I23): e = M_net / R_total
    e_base = safe_div(M_net, R_total, default=0.0)

    # B/6 (Excel I25)
    # Bearing/contact pressure under base (using eccentricity about base center: e_base)
    kern = B / 6.0
    e_eff = e_base  # sign: + -> resultant shifts toward toe (higher toe pressure)

    q_avg = safe_div(R_total, B)  # kPa = (kN/m) / m

    if abs(e_eff) <= kern:
        # Full contact (trapezoidal)
        b_contact = B
        q_toe = q_avg * (1.0 + 6.0 * e_eff / B)
        q_heel = q_avg * (1.0 - 6.0 * e_eff / B)
        sigma_max = max(q_toe, q_heel)
        sigma_min = min(q_toe, q_heel)
        uplift = "none"
    else:
        # Partial contact (triangular), zero tension allowed
        b_contact = max(0.0, 3.0 * (0.5 * B - abs(e_eff)))
        q_max = safe_div(2.0 * R_total, b_contact)  # peak compression
        sigma_max = q_max
        sigma_min = 0.0
        if e_eff > 0:
            # compression at toe, uplift at heel
            q_toe, q_heel = q_max, 0.0
            uplift = "heel"
        else:
            # compression at heel, uplift at toe
            q_toe, q_heel = 0.0, q_max
            uplift = "toe"

    # Stability (Excel I34, I36)
    FS_ot = safe_div(M_stab, M_ot, default=float("nan"))
    FS_sl = safe_div((mu*R_total + Pp), P_H, default=float("nan"))

    return {
        "B":B, "H_active":H_active, "alpha":alpha, "i":i_deg,
        "Ka":Ka, "P_H":P_H, "P_V":P_V,
        "Kp_full":Kp_full, "Kp_reduced":Kp_reduced, "H_front":H_front, "Pp":Pp,
        "df_vertical":df, "R_total":R_total, "M_stab":M_stab,
        "x_R":x_R, "e_load":e_load, "M_ot":M_ot, "M_net":M_net,
        "e_base":e_base, "kern":kern, "b_contact":b_contact,
        "sigma_max":sigma_max, "sigma_min":sigma_min,
        "FS_ot":FS_ot, "FS_sl":FS_sl,
        "q_allow":q_allow, "mu":mu, "beta":beta,
        "L_toe":L_toe, "L_heel":L_heel, "t_f":t_f, "h_stem":h_stem, "t_top":t_top, "t_bot":t_bot,
        "t_cover_front":t_cover_front, "gamma_soil":gamma_soil, "gamma_c":gamma_c,
        "passive_reduction":passive_reduction, "phi":phi, "delta":delta,
    }
