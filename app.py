import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("📈 Courbe G/Gmax en fonction de γ (%)")
st.markdown("Entrez vos paramètres de sol pour générer la courbe G/Gmax avec les incertitudes et les équations empiriques.")

# =============================================
# Équations empiriques
# =============================================
def kollioglou_GGmax(PI, gamma):
    # Constants
    a = 0.99418785
    b = -2.1598671
    c = 10.039495
    d = -16.863967
    e = 0.062926143
    f = -0.013688113
    g = -0.02900694
    h = 5.9454009

    term1 = b * (0.5 + np.arctan((PI - c) / d) / np.pi)
    term2 = e * (0.5 + np.arctan((10**(gamma) - f) / g) / np.pi)
    term3 = h * (0.5 + np.arctan((PI - c) / d) / np.pi)*(0.5 + np.arctan((10**(gamma) - f) / g) / np.pi)

    GGmax = a + term1 + term2 + term3
    return GGmax

def G_over_Gmax_ishibachi(gamma, PI, sigma):
    # Définir les fonctions auxiliaires
    def m0():
        return 0.272

    def m(y, PI):
        term1 = 1 - np.tanh(np.log((0.000556 / y) ** 0.4))
        term2 = np.exp(-0.0145 * PI ** 1.3)
        return m0() * term1 * term2

    def n(PI):
        # Use numpy.where for vectorized conditional logic
        return np.where(PI == 0, 0,
                        np.where((PI > 0) & (PI <= 15), 3.37e-6 * PI ** 1.404,
                                 np.where((PI > 15) & (PI <= 70), 7e-7 * PI ** 1.976,
                                          2.7e-5 * PI ** 1.115))) # PI > 70

    def K(y, PI):
        term1 = 0.5 * (1 + np.tanh(np.log(((0.000102 + n(PI)) / y) ** 0.492)))
        return term1

    # Calculer G/Gmax
    m_value = m(10**(gamma)/100, PI)
    K_value = K(10**(gamma)/100, PI)
    G_Gmax = K_value * sigma ** (m_value)
    return G_Gmax

def vardanega_GGmax(PI, gamma):
    gamma_r = 0.0037 * PI/100
    alpha = 0.943
    GGmax = 1 / (1 + (10**(gamma)/100 / gamma_r)**alpha)
    return GGmax

def G_over_Gmax_ciancimino(gamma, PI, sigma_m_kPa):
    a = 0.9640
    alpha6 = 0.0331
    alpha7 = 0.0014
    alpha8 = 0.1254
    sigma_m_atm = sigma_m_kPa / 101.325
    gamma_r = (alpha6 + alpha7 * PI) * (sigma_m_atm ** alpha8)
    return 1 / (1 + (10**(gamma) / gamma_r) ** a)

def G_over_Gmax_zhang(gamma, PI, sigma_kpa, K0):
    sigma_m = sigma_kpa * (1 + 2 * K0) / 3
    alpha = 0.0021*PI+0.834
    k = 0.316*np.exp(-0.0142*PI)
    gamma_r = (0.0011*PI+0.0749)*(sigma_m/100)**k
    G_Gmax = 1 / (1 + (10**(gamma) / gamma_r) ** alpha)
    return G_Gmax

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
# Paramètres d'entrée dans l'ordre strict du modèle
# =============================================
# Créer des colonnes pour un affichage horizontal
col1, col2, col3, col4 = st.columns(4)

with col1:
     PI = st.number_input("PI", value=20.0, key="PI")
     W = st.number_input("W", value=20.0, key="W")

with col2:
    Z = st.number_input("Z (m)", value=20.0, key="Z")
    sigma = st.number_input("σ (kpa)", value=200.0, key="sigma")  

with col3:
    e0 = st.number_input("e0", value=0.5, key="e0")
    rho = st.number_input("ρ (t/m3)", value=1.5, key="rho")
    
with col4:     
    Wl = st.number_input("Wl", value=20.0, key="Wl")
    K0 = st.number_input("K₀ (pour Zhang)", value=0.5, key="K0")


# Ajouter les paramètres restants
n_points = st.slider("Nombre de points", min_value=5, max_value=20, value=15)

# =============================================
# Logique pour la classe USCS (une seule case cochée)
# =============================================
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
# Options d'affichage et sélection des équations empiriques
# =============================================
st.subheader("Options d'affichage")

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    show_uncertainty = st.checkbox("Afficher les bandes d'incertitude", value=True)

with col_opt2:
    show_model = st.checkbox("Afficher le modèle ML", value=True)

st.subheader("Équations empiriques à comparer")
col_eq1, col_eq2, col_eq3, col_eq4, col_eq5 = st.columns(5)

with col_eq1:
    kollioglou_checked = st.checkbox("Kollioglou", value=False)
with col_eq2:
    ishibashi_checked = st.checkbox("Ishibashi", value=False)
with col_eq3:
    vardanega_checked = st.checkbox("Vardanega", value=False)
with col_eq4:
    ciancimino_checked = st.checkbox("Ciancimino", value=False)
with col_eq5:
    zhang_checked = st.checkbox("Zhang", value=False)

empirical_equations = {
    "Kollioglou": kollioglou_checked,
    "Ishibashi": ishibashi_checked,
    "Vardanega": vardanega_checked,
    "Ciancimino": ciancimino_checked,
    "Zhang": zhang_checked
}

# =============================================
# Calcul et tracé
# =============================================
if st.button("Générer la courbe"):
    gammas = np.logspace(-6, -1, n_points)
    gamma_log = np.log10(gammas)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculer et tracer le modèle ML si demandé
    if show_model:
        X = []
        for g in gammas:
            features = [
                PI, USCS_CH, USCS_CH_CL, USCS_CL, USCS_CL_CH,
                USCS_CL_ML, USCS_MH, USCS_MH_OH, USCS_ML, USCS_ML_OL,
                W, Wl, Z, e0, np.log10(g), rho, sigma
            ]
            X.append(features)
        
        X = np.array(X)
        y_pred = model.predict(X)
        
        # Tracer la courbe principale
        ax.plot(gamma_log, y_pred, label="Modèle ML", color="blue", linewidth=2)
        
        # Ajouter les bandes d'incertitude si demandé
        if show_uncertainty:
            lower_bounds, upper_bounds = get_uncertainty_bounds(y_pred)
            ax.fill_between(gamma_log, lower_bounds, upper_bounds, 
                           alpha=0.3, color="lightblue", label="Bande d'incertitude ML")
            ax.plot(gamma_log, lower_bounds, '--', color="blue", alpha=0.5, linewidth=1)
            ax.plot(gamma_log, upper_bounds, '--', color="blue", alpha=0.5, linewidth=1)
    
    # Calculer et tracer les équations empiriques sélectionnées
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    color_idx = 0
    
    for eq_name, is_selected in empirical_equations.items():
        if is_selected:
            try:
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
                
                # S'assurer que les valeurs sont dans [0, 1]
                y_empirical = np.clip(y_empirical, 0, 1)
                
                ax.plot(gamma_log, y_empirical, 
                       label=f"{eq_name}", color=colors[color_idx], 
                       linewidth=2, linestyle='--')
                color_idx += 1
                
            except Exception as e:
                st.error(f"Erreur lors du calcul de l'équation {eq_name}: {str(e)}")
    
    # Configuration du graphique
    plt.ylim(0, 1)
    ax.set_xlabel("log₁₀(γ) [γ en %]")
    ax.set_ylabel("G/Gmax")
    ax.set_title("Comparaison des courbes de dégradation du module de cisaillement")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajuster la mise en page pour la légende
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Afficher des informations sur les incertitudes si activées
    if show_uncertainty and show_model:
        st.info("""
        **Incertitudes appliquées selon le tableau :**
        - Pour G/Gmax ∈ [0, 0.963] : borne inférieure -0.056, borne supérieure +0.037
        - Pour G/Gmax ∈ [0.963, 1] : borne inférieure -0.056, borne supérieure jusqu'à 1.0
        """)
    
    # Afficher les paramètres utilisés
    with st.expander("Paramètres utilisés"):
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            st.write(f"**Paramètres du sol :**")
            st.write(f"PI = {PI}")
            st.write(f"W = {W}")
            st.write(f"Wl = {Wl}")
        with col_param2:
            st.write(f"**Paramètres géotechniques :**")
            st.write(f"Z = {Z} m")
            st.write(f"σ = {sigma} kPa")
            st.write(f"e₀ = {e0}")
        with col_param3:
            st.write(f"**Autres paramètres :**")
            st.write(f"ρ = {rho} t/m³")
            st.write(f"USCS = {selected_uscs}")
            st.write(f"K₀ = {K0}")
    
        

