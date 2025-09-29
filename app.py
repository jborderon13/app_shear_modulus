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
st.markdown("Enter your soil parameters to generate the G/Gmax curve with uncertainties and empirical equations.")

# =============================================
# Empirical equations
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
# Uncertainty bands
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
# Input parameters
# =============================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    PI = st.number_input("PI", value=20.0)
    W = st.number_input("W", value=20.0)
with col2:
    Z = st.number_input("Z (m)", value=20.0)
    sigma = st.number_input("σ (kPa)", value=200.0)
with col3:
    e0 = st.number_input("e₀", value=0.5)
    rho = st.number_input("ρ (t/m³)", value=1.5)
with col4:
    Wl = st.number_input("Wl", value=20.0)
    K0 = st.number_input("K₀ (for Zhang)", value=0.5)

n_points = st.slider("Number of points", min_value=5, max_value=20, value=15)

uscs_options = ["CH", "CH-CL", "CL", "CL-CH", "CL-ML", "MH", "MH-OH", "ML", "ML-OL"]
selected_uscs = st.radio("Select USCS class:", uscs_options, horizontal=True)

# One-hot encoding
USCS_CH, USCS_CH_CL, USCS_CL = int(selected_uscs=="CH"), int(selected_uscs=="CH-CL"), int(selected_uscs=="CL")
USCS_CL_CH, USCS_CL_ML, USCS_MH = int(selected_uscs=="CL-CH"), int(selected_uscs=="CL-ML"), int(selected_uscs=="MH")
USCS_MH_OH, USCS_ML, USCS_ML_OL = int(selected_uscs=="MH-OH"), int(selected_uscs=="ML"), int(selected_uscs=="ML-OL")

# =============================================
# Display options
# =============================================
st.subheader("Display options")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    show_uncertainty = st.checkbox("Show uncertainty bands", value=True)
with col_opt2:
    show_model = st.checkbox("Show ML model", value=True)

st.subheader("Empirical equations to compare")
col_eq1, col_eq2, col_eq3, col_eq4, col_eq5 = st.columns(5)
kollioglou_checked = col_eq1.checkbox("Kollioglou", value=False)
ishibashi_checked = col_eq2.checkbox("Ishibashi", value=False)
vardanega_checked = col_eq3.checkbox("Vardanega", value=False)
ciancimino_checked = col_eq4.checkbox("Ciancimino", value=False)
zhang_checked = col_eq5.checkbox("Zhang", value=False)

empirical_equations = {
    "Kollioglou": kollioglou_checked,
    "Ishibashi": ishibashi_checked,
    "Vardanega": vardanega_checked,
    "Ciancimino": ciancimino_checked,
    "Zhang": zhang_checked
}

# =============================================
# Computation and plot
# =============================================
if st.button("Generate curve"):
    gammas = np.logspace(-6, -1, n_points)
    gamma_log = np.log10(gammas)

    fig, ax = plt.subplots(figsize=(12, 8))
    results = pd.DataFrame({"log10(gamma)": gamma_log})

    if show_model:
        X = [[PI, USCS_CH, USCS_CH_CL, USCS_CL, USCS_CL_CH,
              USCS_CL_ML, USCS_MH, USCS_MH_OH, USCS_ML, USCS_ML_OL,
              W, Wl, Z, e0, np.log10(g), rho, sigma] for g in gammas]
        X = np.array(X)
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
    ax.set_xlabel("log₁₀(γ) [γ in %]")
    ax.set_ylabel("G/Gmax")
    ax.set_title("Comparison of shear modulus degradation curves")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    st.download_button("💾 Download results as CSV",
                       data=results.to_csv(index=False).encode('utf-8'),
                       file_name="ggmax_results.csv",
                       mime="text/csv")

    with st.expander("Parameters used"):
        st.write(f"PI = {PI}, W = {W}, Wl = {Wl}")
        st.write(f"Z = {Z} m, σ = {sigma} kPa, e₀ = {e0}")
        st.write(f"ρ = {rho} t/m³, USCS = {selected_uscs}, K₀ = {K0}")
```
