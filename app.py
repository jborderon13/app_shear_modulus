import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("📈 Courbe G/Gmax en fonction de γ (%)")
st.markdown("Entrez vos paramètres de sol pour générer la courbe G/Gmax.")

# =============================================
# Paramètres d'entrée dans l'ordre strict du modèle
# =============================================

# Créer des colonnes pour un affichage horizontal
col1, col2, col3, col4 = st.columns(4)

with col1:
     PI = st.number_input("PI", value=20.0, key="PI")
     W = st.number_input("W", value=0.0, key="W")
     Wl = st.number_input("Wl", value=0.0, key="Wl")

with col2:
    rho = st.number_input("ρ (t/m3)", value=0.0, key="rho")
    sigma = st.number_input("σ (kpa)", value=0.0, key="sigma")
    n_points = st.slider("Nombre de points", min_value=10, max_value=200, value=50)
    Z = st.number_input("Z (m)", value=0.0, key="Z")

with col3:
    e0 = st.number_input("e0", value=0.0, key="e0")
    gamma_min = st.number_input("γ min (%)", value=0.01, key="gamma_min")
    gamma_max = st.number_input("γ max (%)", value=1.0, key="gamma_max")



# Ajouter les paramètres restants (ρ et σ) en dessous

# =============================================
# Logique pour la classe USCS (une seule case cochée)
# =============================================
# Désactiver les checkboxes non sélectionnées
uscs_options = ["CH", "CH-CL", "CL", "CL-CH", "CL-ML", "MH", "MH-OH", "ML", "ML-OL"]
selected_uscs = st.radio("Sélectionnez la classe USCS :", uscs_options, horizontal=True)

# Mise à jour des variables one-hot en fonction de la sélection
USCS_CH = 1 if selected_uscs == "CH" else 0
USCS_CH_CL = 1 if selected_uscs == "CH-CL" else 0
USCS_CL = 1 if selected_uscs == "CL" else 0
USCS_CL_CH = 1 if selected_uscs == "CL-CH" else 0
USCS_CL_ML = 1 if selected_uscs == "CL-ML" else 0
USCS_MH = 1 if selected_uscs == "MH" else 0
USCS_MH_OH = 1 if selected_uscs == "MH-OH" else 0
USCS_ML = 1 if selected_uscs == "ML" else 0
USCS_ML_OL = 1 if selected_uscs == "ML-OL" else 0

# =============================================
# Calcul et tracé
# =============================================
if st.button("Générer la courbe"):
    gammas = np.logspace(-8, 1, n_points)
    X = []
    for g in gammas:
        features = [
            PI, USCS_CH, USCS_CH_CL, USCS_CL, USCS_CL_CH,
            USCS_CL_ML, USCS_MH, USCS_MH_OH, USCS_ML, USCS_ML_OL,
            W, Wl, Z, e0, np.log10(g), rho, sigma
        ]
        X.append(features)
    X = np.array(X)
    print(X)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(np.log10(gammas), y_pred, label="G/Gmax", color="blue")
    ax.set_xscale('log')
    ax.set_xlabel("γ (%)")
    ax.set_ylabel("G/Gmax")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
