import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("📈 Courbe G/Gmax en fonction de γ (%)")
st.markdown("Entrez vos paramètres de sol pour générer la courbe G/Gmax avec les incertitudes.")

# =============================================
# Paramètres d'entrée dans l'ordre strict du modèle
# =============================================
# Créer des colonnes pour un affichage horizontal
col1, col2, col3, col4 = st.columns(4)

with col1:
     PI = st.number_input("PI", value=20.0, key="PI")
     W = st.number_input("W", value=20.0, key="W")
     Wl = st.number_input("Wl", value=20.0, key="Wl")

with col2:
    Z = st.number_input("Z (m)", value=20.0, key="Z")
    sigma = st.number_input("σ (kpa)", value=200.0, key="sigma")  

with col3:
    e0 = st.number_input("e0", value=0.5, key="e0")
    rho = st.number_input("ρ (t/m3)", value=1.5, key="rho")

# Ajouter les paramètres restants (ρ et σ) en dessous
n_points = st.slider("Nombre de points", min_value=5, max_value=20, value=15)

# =============================================
# Fonction pour calculer les incertitudes basées sur G/Gmax
# =============================================
def get_uncertainty_bounds(g_gmax_values):
    """
    Calcule les bornes d'incertitude basées sur les valeurs G/Gmax
    selon le tableau : [0,0.963] -> ±0.056, [0.963,1] -> ±0.056
    """
    lower_bounds = []
    upper_bounds = []
    
    for val in g_gmax_values:
        if 0 <= val <= 0.963:
            # Pour la plage [0, 0.963] : -0.056 à +0.037
            lower_bound = max(0, val - 0.056)  # Ne pas descendre en dessous de 0
            upper_bound = min(1, val + 0.037)  # Ne pas dépasser 1
        else:  # val > 0.963
            # Pour la plage [0.963, 1] : -0.056 à +0 (jusqu'à 1)
            lower_bound = max(0, val - 0.056)
            upper_bound = 1.0  # Plafonné à 1
        
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    
    return np.array(lower_bounds), np.array(upper_bounds)

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
# Options d'affichage
# =============================================
show_uncertainty = st.checkbox("Afficher les bandes d'incertitude", value=True)

# =============================================
# Calcul et tracé
# =============================================
if st.button("Générer la courbe"):
    gammas = np.logspace(-6, -1, n_points)
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
    
    # Calculer les incertitudes
    lower_bounds, upper_bounds = get_uncertainty_bounds(y_pred)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer la courbe principale
    ax.plot(np.log10(gammas), y_pred, label="G/Gmax", color="blue", linewidth=2)
    
    # Ajouter les bandes d'incertitude si demandé
    if show_uncertainty:
        ax.fill_between(np.log10(gammas), lower_bounds, upper_bounds, 
                       alpha=0.3, color="lightblue", label="Bande d'incertitude")
        
        # Tracer les bornes
        ax.plot(np.log10(gammas), lower_bounds, '--', color="red", alpha=0.7, linewidth=1)
        ax.plot(np.log10(gammas), upper_bounds, '--', color="red", alpha=0.7, linewidth=1)
    
    plt.ylim(0, 1)
    ax.set_xlabel("log₁₀(γ) [γ en %]")
    ax.set_ylabel("G/Gmax")
    ax.set_title("Courbe de dégradation du module de cisaillement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Ajouter des informations sur les incertitudes
    if show_uncertainty:
        st.info("""
        **Incertitudes appliquées selon le tableau :**
        - Pour G/Gmax ∈ [0, 0.963] : borne inférieure -0.056, borne supérieure +0.037
        - Pour G/Gmax ∈ [0.963, 1] : borne inférieure -0.056, borne supérieure jusqu'à 1.0
        """)
    
    st.pyplot(fig)
    
    # Afficher quelques statistiques
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("G/Gmax min", f"{np.min(y_pred):.3f}")
    with col_stats2:
        st.metric("G/Gmax max", f"{np.max(y_pred):.3f}")
    with col_stats3:
        st.metric("Plage γ", f"10⁻⁶ à 10⁻¹ %")
