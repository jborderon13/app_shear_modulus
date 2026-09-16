from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Shear modulus reduction curves",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------
# Files and model domain
# ---------------------------------------------------------------------
MODEL_PATH = Path("model.pkl")
LOGO_PATH = Path("logo.jpg")

# Range covered by the held-out dataset used to evaluate the empirical
# prediction intervals in the manuscript. Gamma is expressed in percent.
GAMMA_MIN_PCT = 2.30e-5
GAMMA_MAX_PCT = 1.23

# Ranges observed in the curated training/evaluation databases.
OBSERVED_RANGES = {
    "PI": (3.6, 53.0),
    "w": (10.9, 79.6),
    "LL": (25.0, 97.0),
    "Z": (1.5, 75.3),
    "e₀": (0.393, 2.207),
    "ρ": (1.50, 2.23),
    "σ": (40.0, 1100.0),
}


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=1050)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Place the trained model in "
        "the application directory before running the app."
    )
    st.stop()

model = load_model(MODEL_PATH)


st.title("📈 Normalized shear modulus reduction curves")
st.markdown(
    "Enter the soil parameters to predict a normalized shear modulus reduction "
    "curve ($G/G_{max}$) and, if required, compare it with published empirical "
    "formulations. The shaded region represents the nominal 68% empirical "
    "prediction interval derived from held-out residuals."
)
st.info(
    "This application predicts $G/G_{max}$ only. It does not predict damping "
    "and is not a complete site-response model. The point-wise XGBoost model "
    "does not mathematically enforce monotonicity."
)


# =====================================================================
# Empirical formulations
# =====================================================================
def kollioglou_GGmax(PI, gamma_log10_pct):
    a, b, c, d = 0.99418785, -2.1598671, 10.039495, -16.863967
    e, f, g, h = 0.062926143, -0.013688113, -0.02900694, 5.9454009
    p = 0.5 + np.arctan((PI - c) / d) / np.pi
    q = 0.5 + np.arctan((10**gamma_log10_pct - f) / g) / np.pi
    return a + b * p + e * q + h * p * q


def G_over_Gmax_ishibashi(gamma_log10_pct, PI, sigma):
    def m0():
        return 0.272

    def m(y, plasticity_index):
        return (
            m0()
            * (1 - np.tanh(np.log((0.000556 / y) ** 0.4)))
            * np.exp(-0.0145 * plasticity_index**1.3)
        )

    def n(plasticity_index):
        return np.where(
            plasticity_index == 0,
            0.0,
            np.where(
                (plasticity_index > 0) & (plasticity_index <= 15),
                3.37e-6 * plasticity_index**1.404,
                np.where(
                    (plasticity_index > 15) & (plasticity_index <= 70),
                    7e-7 * plasticity_index**1.976,
                    2.7e-5 * plasticity_index**1.115,
                ),
            ),
        )

    def K(y, plasticity_index):
        return 0.5 * (
            1
            + np.tanh(
                np.log(((0.000102 + n(plasticity_index)) / y) ** 0.492)
            )
        )

    strain_fraction = 10**gamma_log10_pct / 100
    return K(strain_fraction, PI) * sigma ** m(strain_fraction, PI)


def vardanega_GGmax(PI, gamma_log10_pct):
    gamma_r = 0.0037 * PI / 100
    alpha = 0.943
    return 1 / (1 + ((10**gamma_log10_pct / 100) / gamma_r) ** alpha)


def G_over_Gmax_ciancimino(gamma_log10_pct, PI, sigma_m_kPa):
    a, alpha6, alpha7, alpha8 = 0.9640, 0.0331, 0.0014, 0.1254
    sigma_m_atm = sigma_m_kPa / 101.325
    gamma_r = (alpha6 + alpha7 * PI) * sigma_m_atm**alpha8
    return 1 / (1 + (10**gamma_log10_pct / gamma_r) ** a)


def G_over_Gmax_zhang(gamma_log10_pct, PI, sigma_kpa, K0=1.0):
    sigma_m = sigma_kpa * (1 + 2 * K0) / 3
    alpha = 0.0021 * PI + 0.834
    k = 0.316 * np.exp(-0.0142 * PI)
    gamma_r = (0.0011 * PI + 0.0749) * (sigma_m / 100) ** k
    return 1 / (1 + (10**gamma_log10_pct / gamma_r) ** alpha)


# =====================================================================
# Strain-dependent empirical prediction intervals
# =====================================================================
def get_prediction_interval(y_pred, gamma_pct):
    """Return the nominal 68% empirical prediction interval.

    Residuals are defined as observed minus predicted. The 16th and 84th
    residual percentiles are applied separately in four shear-strain ranges.
    Only the interval bounds are clipped to the physical range [0, 1].
    """
    y_pred = np.asarray(y_pred, dtype=float)
    gamma_pct = np.asarray(gamma_pct, dtype=float)

    if y_pred.shape != gamma_pct.shape:
        raise ValueError("y_pred and gamma_pct must have identical shapes")

    q16 = np.select(
        [
            gamma_pct < 1e-3,
            gamma_pct < 1e-2,
            gamma_pct < 1e-1,
        ],
        [0.0026, -0.0102, -0.0606],
        default=-0.0655,
    )
    q84 = np.select(
        [
            gamma_pct < 1e-3,
            gamma_pct < 1e-2,
            gamma_pct < 1e-1,
        ],
        [0.0066, 0.0161, 0.0371],
        default=0.0327,
    )

    lower = np.clip(y_pred + q16, 0.0, 1.0)
    upper = np.clip(y_pred + q84, 0.0, 1.0)
    return lower, upper, q16, q84


# =====================================================================
# Inputs
# =====================================================================
def input_with_missing(label, default_value, key, help_text=None):
    value = st.number_input(
        label,
        value=float(default_value),
        key=f"{key}_value",
        help=help_text,
    )
    missing = st.checkbox(
        "Not measured",
        value=False,
        key=f"{key}_missing",
    )
    return np.nan if missing else float(value)


st.subheader("Soil parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    PI = input_with_missing("Plasticity index PI (%)", 20.0, "PI")
    w = input_with_missing("Water content w (%)", 30.0, "w")
with col2:
    Z = input_with_missing("Sample depth Z (m)", 20.0, "Z")
    sigma = input_with_missing(
        "Effective confining pressure σ (kPa)", 200.0, "sigma"
    )
with col3:
    e0 = input_with_missing("Initial void ratio e₀", 0.8, "e0")
    rho = input_with_missing("Density ρ (t/m³)", 1.8, "rho")
with col4:
    LL = input_with_missing("Liquid limit LL (%)", 50.0, "LL")
    K0 = input_with_missing(
        "At-rest earth pressure coefficient K₀",
        1.0,
        "K0",
        help_text=(
            "Used only by the Zhang et al. (2005) formulation. K₀ = 1.0 was "
            "used for the visual comparison reported in the manuscript."
        ),
    )

n_points = st.slider(
    "Number of strain points",
    min_value=10,
    max_value=25,
    value=15,
    step=1,
)

uscs_options = [
    "CH",
    "CH-CL",
    "CL",
    "CL-CH",
    "CL-ML",
    "MH",
    "MH-OH",
    "ML",
    "ML-OL",
    "Not measured",
]
selected_uscs = st.radio(
    "USCS classification",
    uscs_options,
    horizontal=True,
)

uscs_missing = selected_uscs == "Not measured"
if uscs_missing:
    (
        USCS_CH,
        USCS_CH_CL,
        USCS_CL,
        USCS_CL_CH,
        USCS_CL_ML,
        USCS_MH,
        USCS_MH_OH,
        USCS_ML,
        USCS_ML_OL,
    ) = [np.nan] * 9
else:
    USCS_CH = int(selected_uscs == "CH")
    USCS_CH_CL = int(selected_uscs == "CH-CL")
    USCS_CL = int(selected_uscs == "CL")
    USCS_CL_CH = int(selected_uscs == "CL-CH")
    USCS_CL_ML = int(selected_uscs == "CL-ML")
    USCS_MH = int(selected_uscs == "MH")
    USCS_MH_OH = int(selected_uscs == "MH-OH")
    USCS_ML = int(selected_uscs == "ML")
    USCS_ML_OL = int(selected_uscs == "ML-OL")


# ---------------------------------------------------------------------
# Domain and missing-input warnings
# ---------------------------------------------------------------------
model_inputs = {
    "PI": PI,
    "w": w,
    "LL": LL,
    "Z": Z,
    "e₀": e0,
    "ρ": rho,
    "σ": sigma,
}
missing_model_inputs = [name for name, value in model_inputs.items() if np.isnan(value)]
if uscs_missing:
    missing_model_inputs.append("USCS")

if missing_model_inputs:
    st.warning(
        f"Missing model inputs: **{', '.join(missing_model_inputs)}**. XGBoost "
        "will route missing values through learned default branches. However, "
        "systematic feature-masking performance was not evaluated in the study. "
        "Predictions should therefore be interpreted cautiously. The displayed "
        "prediction interval does not include additional uncertainty caused by "
        "missing predictors."
    )

outside_domain = []
for name, value in model_inputs.items():
    if np.isnan(value):
        continue
    lower, upper = OBSERVED_RANGES[name]
    if value < lower or value > upper:
        outside_domain.append(f"{name}={value:g} (observed range: {lower:g}–{upper:g})")

if outside_domain:
    st.warning(
        "Inputs outside the ranges represented in the curated databases: "
        + "; ".join(outside_domain)
        + ". These predictions involve extrapolation and should be interpreted cautiously."
    )


# =====================================================================
# Display options and empirical formulations
# =====================================================================
st.subheader("Display options")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    show_interval = st.checkbox(
        "Show nominal 68% empirical prediction interval",
        value=True,
    )
with col_opt2:
    show_model = st.checkbox("Show XGBoost model", value=True)

empirical_requirements = {
    "Kallioglou (2008)": ["PI"],
    "Ishibashi & Zhang (1993)": ["PI", "σ"],
    "Vardanega & Bolton (2013)": ["PI"],
    "Ciancimino et al. (2020)": ["PI", "σ"],
    "Zhang et al. (2005)": ["PI", "σ", "K₀"],
}
param_values = {"PI": PI, "σ": sigma, "K₀": K0}

st.subheader("Empirical formulations")
eq_cols = st.columns(len(empirical_requirements))
empirical_equations = {}

for (equation_name, required), column in zip(
    empirical_requirements.items(), eq_cols
):
    missing_for_equation = [
        parameter for parameter in required if np.isnan(param_values[parameter])
    ]
    if missing_for_equation:
        column.checkbox(
            equation_name,
            value=False,
            disabled=True,
            help=(
                f"Requires: {', '.join(required)}. "
                f"Missing: {', '.join(missing_for_equation)}."
            ),
            key=f"equation_{equation_name}",
        )
        empirical_equations[equation_name] = False
    else:
        empirical_equations[equation_name] = column.checkbox(
            equation_name,
            value=False,
            help=f"Requires: {', '.join(required)}.",
            key=f"equation_{equation_name}",
        )


# =====================================================================
# Prediction and plot
# =====================================================================
if st.button("Generate curve", type="primary"):
    gamma_pct = np.logspace(
        np.log10(GAMMA_MIN_PCT),
        np.log10(GAMMA_MAX_PCT),
        n_points,
    )
    gamma_log10_pct = np.log10(gamma_pct)

    fig, ax = plt.subplots(figsize=(6, 6))
    results = pd.DataFrame(
        {
            "gamma_percent": gamma_pct,
            "log10_gamma_percent": gamma_log10_pct,
        }
    )

    if show_model:
        # Feature order retained from the trained model pipeline:
        # PI, USCS dummies, w, LL, Z, e0, log10(gamma in %), rho, sigma.
        X = np.array(
            [
                [
                    PI,
                    USCS_CH,
                    USCS_CH_CL,
                    USCS_CL,
                    USCS_CL_CH,
                    USCS_CL_ML,
                    USCS_MH,
                    USCS_MH_OH,
                    USCS_ML,
                    USCS_ML_OL,
                    w,
                    LL,
                    Z,
                    e0,
                    np.log10(gamma),
                    rho,
                    sigma,
                ]
                for gamma in gamma_pct
            ],
            dtype=float,
        )
        y_pred = np.asarray(model.predict(X), dtype=float)
        results["xgboost_prediction"] = y_pred

        ax.plot(
            gamma_log10_pct,
            y_pred,
            label="XGBoost model",
            color="#2166AC",
            linewidth=2.5,
        )

        if np.any((y_pred < 0) | (y_pred > 1)):
            st.warning(
                "At least one central model prediction falls outside the physical "
                "range 0 ≤ G/Gmax ≤ 1. Central predictions are shown without "
                "clipping; only the prediction-interval bounds are constrained."
            )

        if show_interval:
            lower, upper, q16, q84 = get_prediction_interval(y_pred, gamma_pct)
            results["prediction_interval_lower"] = lower
            results["prediction_interval_upper"] = upper
            results["residual_q16"] = q16
            results["residual_q84"] = q84
            ax.fill_between(
                gamma_log10_pct,
                lower,
                upper,
                alpha=0.30,
                color="#92C5DE",
                label="Nominal 68% empirical prediction interval",
            )

    empirical_colors = {
        "Kallioglou (2008)": "#1B9E77",
        "Ishibashi & Zhang (1993)": "#D95F02",
        "Vardanega & Bolton (2013)": "#7570B3",
        "Ciancimino et al. (2020)": "#E7298A",
        "Zhang et al. (2005)": "#A6761D",
    }

    for equation_name, selected in empirical_equations.items():
        if not selected:
            continue

        if equation_name == "Kallioglou (2008)":
            y_empirical = kollioglou_GGmax(PI, gamma_log10_pct)
        elif equation_name == "Ishibashi & Zhang (1993)":
            y_empirical = G_over_Gmax_ishibashi(
                gamma_log10_pct, PI, sigma
            )
        elif equation_name == "Vardanega & Bolton (2013)":
            y_empirical = vardanega_GGmax(PI, gamma_log10_pct)
        elif equation_name == "Ciancimino et al. (2020)":
            y_empirical = G_over_Gmax_ciancimino(
                gamma_log10_pct, PI, sigma
            )
        elif equation_name == "Zhang et al. (2005)":
            y_empirical = G_over_Gmax_zhang(
                gamma_log10_pct, PI, sigma, K0
            )
        else:
            continue

        y_empirical = np.clip(np.asarray(y_empirical, dtype=float), 0.0, 1.0)
        results[equation_name] = y_empirical
        ax.plot(
            gamma_log10_pct,
            y_empirical,
            label=equation_name,
            color=empirical_colors[equation_name],
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlim(np.log10(GAMMA_MIN_PCT), np.log10(GAMMA_MAX_PCT))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\log_{10}(\gamma)$, with $\gamma$ in %", fontsize=16)
    ax.set_ylabel(r"$G/G_{max}$", fontsize=16)
    ax.set_title("Normalized shear modulus reduction curves", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.legend(bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if show_model and show_interval:
        st.caption(
            "The interval uses strain-dependent residual quantiles estimated from "
            "the held-out dataset. Its apparent coverage was 67.6% point-wise and "
            "65.9% test-wise. Because the same held-out residual dataset was used "
            "to estimate the bounds and evaluate coverage, this is not an "
            "independent calibration assessment."
        )

    st.download_button(
        "💾 Download results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="ggmax_results.csv",
        mime="text/csv",
    )

    with st.expander("Parameters used"):
        def format_value(value):
            return "not measured" if np.isnan(value) else f"{value:g}"

        st.write(
            f"PI = {format_value(PI)}%; w = {format_value(w)}%; "
            f"LL = {format_value(LL)}%"
        )
        st.write(
            f"Z = {format_value(Z)} m; σ = {format_value(sigma)} kPa; "
            f"e₀ = {format_value(e0)}"
        )
        st.write(
            f"ρ = {format_value(rho)} t/m³; "
            f"USCS = {'not measured' if uscs_missing else selected_uscs}; "
            f"K₀ = {format_value(K0)} (Zhang et al. only)"
        )
        st.write(
            f"Prediction range: {GAMMA_MIN_PCT:.2e}% ≤ γ ≤ "
            f"{GAMMA_MAX_PCT:g}%"
        )
