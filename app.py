import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load logo
st.image("logo.jpg", width=1050)

# Load model
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("📈 G/Gmax curve as a function of γ (%)")
st.markdown(
    "Enter your soil parameters to generate the G/Gmax curve with uncertainties and empirical equations. "
    "If a parameter has not been measured, check the corresponding **Not measured** box: "
    "the XGBoost model handles missing inputs natively, while empirical equations requiring "
    "that parameter will be automatically disabled."
)

# =============================================
# Empirical equations (unchanged)
# =============================================
def kollioglou_GGmax(PI, gamma):
    a, b, c, d = 0.99418785, -2.1598671, 10.039495, -16.863967
    e, f, g, h = 0.062926143, -0.013688113, -0.02900694, 5.9454009
    term1 = b * (0.5 + np.arctan((PI - c) / d) / np.pi)
    term2 = e * (0.5 + np.arctan((10**(gamma) - f) / g) / np.pi)
    term3 = h * (0.5 + np.arctan((PI - c) / d) / np.pi) * (0.5 + np.arctan((10**(gamma) - f) / g) / np.pi)
    return a + term1 + term2 + term3

def G_over_Gmax_ishibachi(gamma, PI, sigma):
    def m0(): return 0.272
    def m(y, PI):
        return m0() * (1 - np.tanh(np.log((0.000556 / y) ** 0.4))) * np.exp(-0.0145 * PI ** 1.3)
    def n(PI):
        return np.where(PI == 0, 0,
                        np.where((PI > 0) & (PI <= 15), 3.37e-6 * PI ** 1.404,
                                 np.where((PI > 15) & (PI <= 70), 7e-7 * PI ** 1.976,
                                          2.7e-5 * PI ** 1.115)))
    def K(y, PI):
        return 0.5 * (1 + np.tanh(np.log(((0.000102 + n(PI)) / y) ** 0.492)))
    return K(10**(gamma)/100, PI) * sigma ** (m(10**(gamma)/100, PI))

def vardanega_GGmax(PI, gamma):
    gamma_r = 0.0037 * PI / 100
    alpha = 0.943
    return 1 / (1 + (10**(gamma)/100 / gamma_r) ** alpha)

def G_over_Gmax_ciancimino(gamma, PI, sigma_m_kPa):
    a, alpha6, alpha7, alpha8 = 0.9640, 0.0331, 0.0014, 0.1254
    sigma_m_atm = sigma_m_kPa / 101.325
    gamma_r = (alpha6 + alpha7 * PI) * (sigma_m_atm ** alpha8)
    return 1 / (1 + (10**(gamma) / gamma_r) ** a)

def G_over_Gmax_zhang(gamma, PI, sigma_kpa, K0):
    sigma_m = sigma_kpa * (1 + 2 * K0) / 3
    alpha = 0.0021 * PI + 0.834
    k = 0.316 * np.exp(-0.0142 * PI)
    gamma_r = (0.0011 * PI + 0.0749) * (sigma_m / 100) ** k
    return 1 / (1 + (10**(gamma) / gamma_r) ** alpha)

# =============================================
# Uncertainty bands (unchanged)
# =============================================
def get_uncertainty_bounds(g_gmax_values):
    lower_bounds, upper_bounds = [], []
    for val in g_gmax_values:
        if 0 <= val <= 0.963:
            lower_bounds.append(max(0, val - 0.056))
            upper_bounds.append(min(1, val + 0.037))
        else:
            lower_bounds.append(max(0, val - 0.056))
            upper_bounds.append(1.0)
    return np.array(lower_bounds), np.array(upper_bounds)

# =============================================
# Helper: input with "Not measured" checkbox
# =============================================
def input_with_missing(label, default_value, key):
    """
    Returns the value as float, or np.nan if 'Not measured' is checked.
    """
    val = st.number_input(label, value=default_value, key=f"{key}_val")
    missing = st.checkbox("Not measured", value=False, key=f"{key}_missing")
    return np.nan if missing else val

# =============================================
# Input parameters
# =============================================
st.subheader("Soil parameters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    PI = input_with_missing("PI", 20.0, "PI")
    W = input_with_missing("W", 20.0, "W")
with col2:
    Z = input_with_missing("Z (m)", 20.0, "Z")
    sigma = input_with_missing("σ (kPa)", 200.0, "sigma")
with col3:
    e0 = input_with_missing("e₀", 0.5, "e0")
    rho = input_with_missing("ρ (t/m³)", 1.5, "rho")
with col4:
    Wl = input_with_missing("Wl", 20.0, "Wl")
    K0 = input_with_missing("K₀ (for Zhang)", 0.5, "K0")

n_points = st.slider("Number of points", min_value=5, max_value=20, value=15)

uscs_options = ["CH", "CH-CL", "CL", "CL-CH", "CL-ML", "MH", "MH-OH", "ML", "ML-OL", "Not measured"]
selected_uscs = st.radio("Select USCS class:", uscs_options, horizontal=True)

# One-hot encoding (all zeros if "Not measured" is selected)
uscs_missing = (selected_uscs == "Not measured")
USCS_CH    = int(selected_uscs == "CH")
USCS_CH_CL = int(selected_uscs == "CH-CL")
USCS_CL    = int(selected_uscs == "CL")
USCS_CL_CH = int(selected_uscs == "CL-CH")
USCS_CL_ML = int(selected_uscs == "CL-ML")
USCS_MH    = int(selected_uscs == "MH")
USCS_MH_OH = int(selected_uscs == "MH-OH")
USCS_ML    = int(selected_uscs == "ML")
USCS_ML_OL = int(selected_uscs == "ML-OL")

# =============================================
# Summary of missing inputs
# =============================================
missing_params = []
if np.isnan(PI):    missing_params.append("PI")
if np.isnan(W):     missing_params.append("W")
if np.isnan(Wl):    missing_params.append("Wl")
if np.isnan(Z):     missing_params.append("Z")
if np.isnan(sigma): missing_params.append("σ")
if np.isnan(e0):    missing_params.append("e₀")
if np.isnan(rho):   missing_params.append("ρ")
if np.isnan(K0):    missing_params.append("K₀")
if uscs_missing:    missing_params.append("USCS")

if missing_params:
    st.warning(
        f"⚠️ Missing inputs: **{', '.join(missing_params)}**. "
        "The XGBoost model will handle them natively (with potentially increased uncertainty). "
        "Empirical equations requiring any of these parameters are disabled below."
    )

# =============================================
# Display options
# =============================================
st.subheader("Display options")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    show_uncertainty = st.checkbox("Show uncertainty bands", value=True)
with col_opt2:
    show_model = st.checkbox("Show ML model", value=True)

# =============================================
# Empirical equations: dependencies + dynamic enabling
# =============================================
empirical_requirements = {
    "Kollioglou": ["PI"],
    "Ishibashi": ["PI", "σ"],
    "Vardanega":  ["PI"],
    "Ciancimino": ["PI", "σ"],
    "Zhang":      ["PI", "σ", "K₀"],
}

param_values = {"PI": PI, "σ": sigma, "K₀": K0}

st.subheader("Empirical equations to compare")
eq_cols = st.columns(len(empirical_requirements))
empirical_equations = {}

for (eq_name, required), col in zip(empirical_requirements.items(), eq_cols):
    missing_for_eq = [p for p in required if np.isnan(param_values[p])]
    if missing_for_eq:
        col.checkbox(
            f"{eq_name}",
            value=False,
            disabled=True,
            help=f"Requires: {', '.join(required)}. Missing: {', '.join(missing_for_eq)}.",
            key=f"eq_{eq_name}",
        )
        empirical_equations[eq_name] = False
    else:
        empirical_equations[eq_name] = col.checkbox(
            f"{eq_name}",
            value=False,
            help=f"Requires: {', '.join(required)}.",
            key=f"eq_{eq_name}",
        )

# =============================================
# Computation and plot
# =============================================
if st.button("Generate curve"):
    gammas = np.logspace(-6, -1, n_points)
    gamma_log = np.log10(gammas)

    fig, ax = plt.subplots(figsize=(12, 8))
    results = pd.DataFrame({"log10(gamma)": gamma_log})

    if show_model:
        # Build feature matrix: NaN propagates naturally to XGBoost
        X = np.array([
            [PI, USCS_CH, USCS_CH_CL, USCS_CL, USCS_CL_CH,
             USCS_CL_ML, USCS_MH, USCS_MH_OH, USCS_ML, USCS_ML_OL,
             W, Wl, Z, e0, np.log10(g), rho, sigma]
            for g in gammas
        ], dtype=float)
        y_pred = model.predict(X)
        results["ML_Model"] = y_pred

        ax.plot(gamma_log, y_pred, label="ML Model", color="blue", linewidth=2)
        if show_uncertainty:
            lower_bounds, upper_bounds = get_uncertainty_bounds(y_pred)
            results["ML_Lower"] = lower_bounds
            results["ML_Upper"] = upper_bounds
            ax.fill_between(gamma_log, lower_bounds, upper_bounds, alpha=0.3,
                            color="lightblue", label="ML Uncertainty Band")

    colors = ['red', 'green', 'orange', 'purple', 'brown']
    color_idx = 0
    for eq_name, is_selected in empirical_equations.items():
        if is_selected:
            if eq_name == "Kollioglou":
                y_empirical = kollioglou_GGmax(PI, gamma_log)
            elif eq_name == "Ishibashi":
                y_empirical = G_over_Gmax_ishibachi(gamma_log, PI, sigma)
            elif eq_name == "Vardanega":
                y_empirical = vardanega_GGmax(PI, gamma_log)
            elif eq_name == "Ciancimino":
                y_empirical = G_over_Gmax_ciancimino(gamma_log, PI, sigma)
            elif eq_name == "Zhang":
                y_empirical = G_over_Gmax_zhang(gamma_log, PI, sigma, K0)
            y_empirical = np.clip(y_empirical, 0, 1)
            results[eq_name] = y_empirical
            ax.plot(gamma_log, y_empirical, label=eq_name,
                    color=colors[color_idx], linewidth=2, linestyle='--')
            color_idx += 1

    ax.set_ylim(0, 1)
    ax.set_xlabel("log₁₀(γ) [γ in %]", fontsize=18)
    ax.set_ylabel("G/Gmax", fontsize=18)
    ax.set_title("Comparison of shear modulus degradation curves", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 16})

    plt.tight_layout()
    st.pyplot(fig)

    st.download_button("💾 Download results as CSV",
                       data=results.to_csv(index=False).encode('utf-8'),
                       file_name="ggmax_results.csv",
                       mime="text/csv")

    with st.expander("Parameters used"):
        def fmt(v): return "not measured" if (isinstance(v, float) and np.isnan(v)) else v
        st.write(f"PI = {fmt(PI)}, W = {fmt(W)}, Wl = {fmt(Wl)}")
        st.write(f"Z = {fmt(Z)} m, σ = {fmt(sigma)} kPa, e₀ = {fmt(e0)}")
        st.write(f"ρ = {fmt(rho)} t/m³, USCS = {'not measured' if uscs_missing else selected_uscs}, K₀ = {fmt(K0)}")
        if missing_params:
            st.info(f"Missing inputs handled natively by XGBoost: {', '.join(missing_params)}.")
