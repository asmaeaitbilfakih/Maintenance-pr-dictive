"""
Maintenance prédictive : interface de supervision (Streamlit)
Lancer depuis la racine du projet :  streamlit run app/app.py
"""
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- configuration
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
AUTHORS = "Asmae Ait Bilfakih & Asma Azizi"

PRIMARY, OK, WARN, ALERT = "#1F4E79", "#2F7D4F", "#B7791F", "#B42318"
INK, MUTED, GRAY, LINE = "#1F2933", "#52606D", "#9AA5B1", "#D9DEE4"
ACCENT = BLUE = PRIMARY
STEEL = GRAY

FEATS = ["Type", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
         "temp_diff", "power_w", "wear_torque"]
FEAT_FR = {
    "Type": "Classe de l'équipement", "air_temp": "Température ambiante",
    "process_temp": "Température process", "rpm": "Vitesse de rotation",
    "torque": "Couple", "tool_wear": "Usure de la pièce",
    "temp_diff": "Écart thermique ΔT", "power_w": "Puissance mécanique",
    "wear_torque": "Usure × couple",
}
TYPE_LABEL = {0: "L", 1: "M", 2: "H"}

st.set_page_config(page_title="Maintenance prédictive · Cimenterie", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1280px; }
.page-h { border-bottom: 1px solid #D9DEE4; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.page-h h1 { margin: 0 0 .35rem 0; padding: 0; font-size: 1.7rem; font-weight: 600; color: #1F2933; letter-spacing: -.01em; }
.page-h p { margin: 0; color: #52606D; font-size: .97rem; max-width: 780px; line-height: 1.55; }
.kpi { background: #FFFFFF; border: 1px solid #D9DEE4; border-left: 3px solid var(--c, #D9DEE4); border-radius: 6px; padding: .85rem 1rem; height: 100%; margin-bottom: 1.4rem; }
.kpi-l { font-size: .82rem; color: #52606D; }
.kpi-v { font-size: 1.7rem; font-weight: 600; color: #1F2933; line-height: 1.3; font-variant-numeric: tabular-nums; }
.kpi-s { font-size: .8rem; color: #7B8794; }
.badge { display: inline-flex; align-items: center; gap: .4rem; padding: .18rem .6rem; border-radius: 4px;
         border: 1px solid var(--c); color: var(--c); font-weight: 600; font-size: .82rem;
         background: color-mix(in srgb, var(--c) 8%, #FFFFFF); }
.badge i { width: .5rem; height: .5rem; border-radius: 50%; background: var(--c); }
.mc { background: #FFFFFF; border: 1px solid #D9DEE4; border-top: 3px solid var(--c); border-radius: 6px; padding: .75rem .9rem; margin-bottom: .8rem; }
.mc-top { display: flex; justify-content: space-between; align-items: center; }
.tag { font-weight: 600; color: #1F2933; font-size: .95rem; font-variant-numeric: tabular-nums; }
.led { width: .6rem; height: .6rem; border-radius: 50%; background: var(--c); }
.mc-name { color: #52606D; font-size: .8rem; min-height: 2.1em; }
.mc-risk { font-size: 1.8rem; font-weight: 600; color: var(--n, var(--c)); font-variant-numeric: tabular-nums; }
.mc-risk small { font-size: .85rem; font-weight: 400; color: #7B8794; }
.bar { background: #E5E9EE; border-radius: 3px; height: 5px; overflow: hidden; margin: .25rem 0 .5rem 0; }
.bar div { height: 100%; background: var(--c); }
.mc-meta { color: #7B8794; font-size: .74rem; font-variant-numeric: tabular-nums; }
.chk { background: #FFFFFF; border: 1px solid #D9DEE4; border-radius: 6px; padding: .7rem .9rem; height: 100%; }
.chk-h { display: flex; justify-content: space-between; align-items: center; margin-bottom: .35rem; }
.chk-n { color: #1F2933; font-weight: 500; font-size: .9rem; }
.chk-m { color: #52606D; font-size: .78rem; margin-top: .35rem; font-variant-numeric: tabular-nums; }
.chk-a { color: #1F2933; font-size: .8rem; margin-top: .4rem; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .4rem 0 1.4rem 0; }
.step { background: #FFFFFF; border: 1px solid #D9DEE4; border-radius: 6px; padding: .5rem .8rem; color: #52606D; font-size: .86rem; }
.step.hl { border-color: #1F4E79; color: #1F4E79; font-weight: 500; background: #EEF3F8; }
.arrow { color: #9AA5B1; }
.card { background: #FFFFFF; border: 1px solid #D9DEE4; border-radius: 6px; padding: 1rem 1.2rem; height: 100%; }
.card.hl { border-color: #1F4E79; border-width: 2px; }
.card h4 { margin: 0 0 .4rem 0; padding: 0; color: #1F2933; font-weight: 600; font-size: 1.05rem; }
.card p { margin: 0; color: #52606D; font-size: .9rem; line-height: 1.5; }
.action { background: #EEF3F8; border: 1px solid #D9DEE4; border-left: 3px solid #1F4E79; border-radius: 6px; padding: .7rem 1rem; color: #1F2933; font-size: .92rem; }
.foot { color: #7B8794; font-size: .78rem; line-height: 1.5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- utilitaires
def fr(x, d=0):
    """Format français : espace fine pour les milliers, virgule décimale."""
    return f"{x:,.{d}f}".replace(",", "\u202f").replace(".", ",")


def badge(label, color):
    return f'<span class="badge" style="--c:{color}"><i></i>{label}</span>'


def kpi(label, value, sub="", color=None):
    style = f' style="--c:{color}"' if color else ""
    st.markdown(f'<div class="kpi"{style}><div class="kpi-l">{label}</div>'
                f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>',
                unsafe_allow_html=True)


def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="page-h"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def show(fig, height=380):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="IBM Plex Sans, Segoe UI, sans-serif"), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#E5E9EE", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#E5E9EE", zeroline=False, linecolor=LINE)
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_image(name, caption=None):
    path = FIG / name
    if not path.exists():
        st.info(f"Figure introuvable : {name}")
        return
    try:
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        st.image(str(path), caption=caption, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)



# --------------------------------------------------------------------------- valeurs de secours
# Utilisées seulement si un CSV de reports/tables est absent ou n'a pas le bon format.
FALLBACK_CSV = {
"comparaison_approches.csv": """Approche,Étiquettes nécessaires,PR-AUC (panne < 48 h),MAD/h,économie vs corrective
Corrective (attendre la panne),—,—,457.0997580771119,0%
Survie : remplacement à âge fixe,oui (durées de vie),—,297.86456794614344,35%
Anomalies (One-Class SVM),non,0.576,106.2802361702596,77%
"Classifieur (XGBoost, seuil coût)",oui (« panne dans 48 h »),0.892,79.1544037534858,83%
Idéale (limite théorique),—,—,77.89008210647759,83%
""",
"resultats_anomalies.csv": """modèle,ROC-AUC,PR-AUC,seuil,pannes détectées,taux de détection,délai médian (h),fausses alarmes (sain)
Isolation Forest,0.941,0.314,0.522,43/44,0.977,1038.0,0.0088
One-Class SVM,0.983,0.576,-1.796,43/44,0.977,667.0,0.0018
Autoencodeur,0.955,0.391,2.2,43/44,0.977,592.0,0.0
""",
"resultats_classification.csv": """Modèle,PR-AUC,ROC-AUC,Recall,Précision,F1
Régression logistique,0.403,0.932,0.838,0.17,0.283
Random Forest,0.899,0.987,0.838,0.983,0.905
XGBoost,0.861,0.989,0.853,0.716,0.779
""",
"resultats_classification_synthetique.csv": """Modèle,PR-AUC (CV train),PR-AUC (test),ROC-AUC (test),Recall,Précision,F1
Régression logistique,0.868,0.834,0.997,0.98,0.453,0.619
Random Forest,0.85,0.883,0.998,0.92,0.695,0.792
XGBoost,0.866,0.892,0.998,0.966,0.633,0.765
""",
"resultats_cout_classifieur.csv": """,MAD/h,MAD/an/machine,économie vs corrective
Corrective (attendre la panne),457.0997580771119,3657000.0,0%
Préventive à âge fixe,297.86456794614344,2383000.0,35%
Prédictive (alarme anomalies),106.2802361702596,850000.0,77%
"Prédictive (classifieur, seuil 0,5)",79.4049102204833,635000.0,83%
"Prédictive (classifieur, seuil coût = 0.8)",79.1544037534858,633000.0,83%
Idéale (limite théorique),77.89008210647759,623000.0,83%
""",
"resultats_cout_sensibilite.csv": """facteur perte prod,corrective,fixe,alarme,idéale
0.1,83.34,54.36,17.81,12.79
0.25,145.64,97.09,32.56,23.64
0.5,249.46,164.94,57.13,41.72
1.0,457.1,297.86,106.28,77.89
2.0,872.38,566.05,204.58,150.22
4.0,1702.95,1103.51,401.18,294.89
""",
"resultats_survie.csv": """type,forme ρ estimée,IC95 % ρ,forme vraie (beta),échelle λ estimée (h),"échelle vraie (eta, h)",âge optimal de remplacement T* (h),coût/h à T*,coût/h sans préventif,économie
broyeur,1.77,[1.47 ; 2.07],2.2,2196.06,1800,860,0.00275,0.00409,33%
compresseur,2.36,[1.86 ; 2.85],3.0,3440.04,3000,1330,0.00133,0.00262,49%
ventilateur,2.33,[1.86 ; 2.81],2.6,3137.05,2400,1210,0.00147,0.00288,49%
""",
"shap_importance.csv": """,mean_abs_shap
vib_rms_ratio_baseline,1.6887242
vib_rms_slope_12,1.3168464
vib_kurtosis_mean_4,1.2326026
temperature_ratio_baseline,0.8112209
vib_crest_mean_4,0.7928685
current_ratio_baseline,0.7459377
vib_rms_mean_4,0.6812528
vib_kurtosis,0.61235124
temperature_slope_12,0.54707295
temperature_mean_4,0.48061454
vib_kurtosis_ratio_baseline,0.4394096
temperature,0.42751878
current_slope_12,0.42555052
vib_crest,0.4134905
pressure_mean_4,0.3917299
""",
}


def read_table(name, **kw):
    """Lit reports/tables/<name> ; bascule sur les valeurs de secours si le fichier manque ou diffère."""
    fb = pd.read_csv(io.StringIO(FALLBACK_CSV[name]), **kw)
    try:
        df = pd.read_csv(TAB / name, **kw)
        if list(df.columns) == list(fb.columns) and len(df) > 0:
            return df
    except Exception:
        pass
    return fb


def pretty_feature(name):
    sensors = {"vib_rms": "Vibration RMS", "vib_kurtosis": "Kurtosis vibration",
               "vib_crest": "Facteur de crête", "temperature": "Température",
               "pressure": "Pression", "current": "Courant moteur"}
    suffixes = {"_ratio_baseline": " · écart à l'état sain", "_slope_12": " · tendance 72 h",
                "_mean_4": " · moyenne 24 h", "_std_4": " · variabilité 24 h"}
    for suf, txt in suffixes.items():
        if name.endswith(suf):
            return sensors.get(name[: -len(suf)], name) + txt
    return sensors.get(name, name) + " · mesure brute"


# --------------------------------------------------------------------------- chargement
@st.cache_data
def load_ai4i():
    path = ROOT / "data" / "ai4i2020.csv"
    if not path.exists():
        st.error(f"Fichier introuvable : {path}\nPlacez ai4i2020.csv dans le dossier data/.")
        st.stop()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def ai4i_features():
    """Reconstruit les 9 features du notebook 01 à partir du CSV brut (températures en °C)."""
    ai = load_ai4i()
    raw = pd.DataFrame({
        "Type": ai["Type"].map({"L": 0, "M": 1, "H": 2}),
        "air_temp": ai["Air temperature [K]"] - 273.15,
        "process_temp": ai["Process temperature [K]"] - 273.15,
        "rpm": ai["Rotational speed [rpm]"],
        "torque": ai["Torque [Nm]"],
        "tool_wear": ai["Tool wear [min]"],
    })
    raw["temp_diff"] = raw["process_temp"] - raw["air_temp"]
    raw["power_w"] = raw["torque"] * raw["rpm"] * 2 * np.pi / 60
    raw["wear_torque"] = raw["tool_wear"] * raw["torque"]
    return raw[FEATS]


@st.cache_resource
def load_assets():
    files = {"XGBoost": "xgboost.pkl", "Random Forest": "random_forest.pkl", "Régression logistique": "logreg.pkl"}
    missing = [f for f in files.values() if not (ROOT / "models" / f).exists()]
    if missing:
        st.error(f"Modèles introuvables dans {ROOT / 'models'} : {', '.join(missing)}")
        st.stop()
    models = {name: joblib.load(ROOT / "models" / f) for name, f in files.items()}
    for m in models.values():  # compat. : LogisticRegression sauvegardée sans l'attribut multi_class
        if type(m).__name__ == "LogisticRegression" and not hasattr(m, "multi_class"):
            m.multi_class = "auto"
    sc_path = ROOT / "data" / "processed" / "scaler.pkl"
    if sc_path.exists():
        return models, joblib.load(sc_path), False
    from sklearn.preprocessing import StandardScaler  # secours : scaler approché
    return models, StandardScaler().fit(ai4i_features()), True


MODELS, SCALER, SCALER_APPROX = load_assets()


def build_row(type_idx, air, proc, rpm, torque, wear):
    d = dict(Type=type_idx, air_temp=air, process_temp=proc, rpm=rpm, torque=torque, tool_wear=wear)
    d["temp_diff"] = proc - air
    d["power_w"] = torque * rpm * 2 * np.pi / 60
    d["wear_torque"] = wear * torque
    return pd.DataFrame([d])[FEATS]


def scale(raw):
    return pd.DataFrame(SCALER.transform(raw[FEATS]), columns=FEATS)


def predict(raw, model_name):
    return float(MODELS[model_name].predict_proba(scale(raw))[:, 1][0])


def local_contribs(raw):
    booster = MODELS["XGBoost"].get_booster()
    c = booster.predict(xgb.DMatrix(scale(raw)), pred_contribs=True)[0][:-1]
    return pd.Series(c, index=FEATS)


LIM_OSF = {0: 11000, 1: 12000, 2: 13000}


def physical_checks(r):
    """Règles physiques des 4 modes de défaillance (documentation du jeu AI4I 2020)."""
    td, pw, wt = r.process_temp - r.air_temp, r.power_w, r.wear_torque
    lim = LIM_OSF[int(r.Type)]
    return [
        ("TWF", "Usure de la pièce", r.tool_wear / 200,
         f"{r.tool_wear:.0f} min · zone critique ≥ 200 min",
         "Remplacer la pièce d'usure (outil, garniture, revêtement)."),
        ("HDF", "Dissipation thermique", min(8.6 / max(td, 0.1), 1380 / max(r.rpm, 1)),
         f"ΔT {td:.1f} °C (< 8,6) · {r.rpm:.0f} tr/min (< 1 380)",
         "Contrôler la ventilation et le refroidissement des paliers et moteurs."),
        ("PWF", "Puissance mécanique", max(pw / 9000, 3500 / max(pw, 1)),
         f"{pw / 1000:.1f} kW · plage normale 3,5 à 9,0 kW",
         "Vérifier l'alimentation moteur, la charge et l'accouplement."),
        ("OSF", "Surcharge mécanique", wt / lim,
         f"{fr(wt)} min·N·m · limite {fr(lim)}",
         "Réduire la charge, contrôler le réducteur et l'alignement."),
    ]


def check_status(ratio):
    if ratio >= 1:
        return "Déclenché", ALERT
    if ratio >= 0.92:
        return "Proche", WARN
    return "OK", OK


def risk_status(p, thr):
    watch = max(0.05, thr * 0.4)
    if p >= thr:
        return "Alerte", ALERT, "Arrêt contrôlé et intervention sous 24 h : inspecter les modes de défaillance signalés."
    if p >= watch:
        return "Surveillance", WARN, "Renforcer la surveillance (relevés toutes les 6 h) et préparer les pièces de rechange."
    return "Normal", OK, "Fonctionnement normal : poursuivre la maintenance planifiée."


# --------------------------------------------------------------------------- bloc diagnostic (réutilisé)
def render_diagnostic(raw, model_name, thr):
    r = raw.iloc[0]
    p = predict(raw, model_name)
    label, color, action = risk_status(p, thr)
    watch = max(0.05, thr * 0.4)

    c1, c2 = st.columns([1, 1.25])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number=dict(suffix=" %", font=dict(size=44, color=color)),
            title=dict(text="Risque de panne", font=dict(size=14, color=MUTED)),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=GRAY), bar=dict(color=color, thickness=0.28),
                       bgcolor="#F3F5F8", borderwidth=0,
                       steps=[dict(range=[0, watch * 100], color="rgba(47,125,79,.18)"),
                              dict(range=[watch * 100, thr * 100], color="rgba(183,121,31,.20)"),
                              dict(range=[thr * 100, 100], color="rgba(180,35,24,.18)")],
                       threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=thr * 100))))
        show(fig, 270)
    with c2:
        st.markdown(badge(label, color), unsafe_allow_html=True)
        st.markdown(f'<div class="action" style="margin:.6rem 0 .8rem 0">{action}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, name in zip(cols, MODELS):
            with col:
                pm = predict(raw, name)
                kpi(name, f"{pm * 100:.0f} %", "score du modèle", ALERT if pm >= thr else (WARN if pm >= watch else OK))

    st.markdown("##### Contrôles physiques des modes de défaillance")
    checks = physical_checks(r)
    cols = st.columns(4)
    for col, (code, name, ratio, meas, act) in zip(cols, checks):
        st_label, st_color = check_status(ratio)
        with col:
            extra = f'<div class="chk-a">{act}</div>' if st_label != "OK" else ""
            st.markdown(
                f'<div class="chk" style="border-top:3px solid {st_color}"><div class="chk-h">'
                f'<span class="tag">{code}</span>{badge(st_label, st_color)}</div>'
                f'<div class="chk-n">{name}</div><div class="bar"><div style="width:{min(ratio, 1) * 100:.0f}%;background:{st_color}"></div></div>'
                f'<div class="chk-m">{meas}</div>{extra}</div>', unsafe_allow_html=True)

    st.markdown("")
    c3, c4 = st.columns(2)
    with c3:
        contrib = local_contribs(raw)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6][::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[FEAT_FR[i] for i in top.index], orientation="h",
                               marker_color=[ALERT if v > 0 else PRIMARY for v in top.values]))
        fig.update_layout(title="Facteurs qui pèsent sur le score (XGBoost)",
                          xaxis_title="← réduit le risque · augmente le risque →")
        show(fig, 320)
    with c4:
        z = scale(raw).iloc[0].drop("Type")
        fig = go.Figure(go.Bar(x=z.values, y=[FEAT_FR[i] for i in z.index], orientation="h",
                               marker_color=[ALERT if abs(v) > 2 else GRAY for v in z.values]))
        fig.add_vline(x=-2, line_dash="dot", line_color=STEEL)
        fig.add_vline(x=2, line_dash="dot", line_color=STEEL)
        fig.update_layout(title="Écart à la moyenne du parc (en écarts-types)", xaxis_title="σ")
        show(fig, 320)
    return p


# --------------------------------------------------------------------------- flotte simulée
EQUIP = [
    ("VT-101", "Ventilateur de tirage · four"), ("VT-102", "Ventilateur de tirage · filtre"),
    ("VT-103", "Ventilateur · refroidisseur clinker"), ("BR-201", "Broyeur cru · moteur principal"),
    ("BR-202", "Broyeur cru · réducteur"), ("BR-203", "Broyeur ciment · moteur principal"),
    ("BR-204", "Broyeur ciment · séparateur"), ("CA-301", "Compresseur d'air · ensachage"),
    ("CA-302", "Compresseur d'air · filtres à manches"), ("CA-303", "Compresseur d'air · instrumentation"),
    ("VT-104", "Ventilateur d'extraction · concasseur"), ("BR-205", "Broyeur à charbon"),
]


@st.cache_data
def build_fleet():
    raw = ai4i_features().reset_index(drop=True)
    p = MODELS["XGBoost"].predict_proba(scale(raw))[:, 1]
    rng = np.random.default_rng(11)

    def pick(mask, k):
        idx = np.where(mask)[0]
        return list(rng.choice(idx, min(k, len(idx)), replace=False))

    chosen = pick(p < 0.03, 7) + pick((p >= 0.1) & (p < 0.5), 3) + pick(p >= 0.9, 2)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    fleet = raw.iloc[chosen].reset_index(drop=True)
    fleet["Type"] = fleet["Type"].astype(int)
    fleet["tag"] = [e[0] for e in EQUIP[: len(fleet)]]
    fleet["name"] = [e[1] for e in EQUIP[: len(fleet)]]
    return fleet


# --------------------------------------------------------------------------- sidebar
PAGES = ["Vue d'ensemble", "Supervision du parc", "Diagnostic d'un équipement",
         "Fiabilité et cycle de vie", "Analyse économique", "Modèles et explicabilité"]

with st.sidebar:
    st.markdown('<div style="font-size:1.2rem;font-weight:600;color:#1F2933">'
                'Maintenance prédictive</div><div class="foot">Cimenterie, équipements tournants</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Paramètres d'alerte**")
    MODEL_NAME = st.selectbox("Modèle de prédiction", list(MODELS), index=0)
    THR = st.slider("Seuil d'alerte (risque de panne)", 0.10, 0.90, 0.50, 0.05)
    if SCALER_APPROX:
        st.warning("scaler.pkl introuvable (data/processed) : normalisation approchée.")
    st.markdown("---")
    st.markdown(f'<div class="foot">Données : AI4I 2020 (UCI) + cimenterie simulée<br>Projet de {AUTHORS}</div>',
                unsafe_allow_html=True)


# =========================================================================== PAGE 1
def page_overview():
    header("SUPERVISION CONDITIONNELLE · INDUSTRIE 4.0", "Prédire la panne avant qu'elle arrive",
           "Surveillance des ventilateurs de tirage, broyeurs et compresseurs d'air d'une cimenterie : "
           "vibrations, température, pression et courant pour intervenir au bon moment.")

    ai4i = load_ai4i()
    clf = read_table("resultats_classification.csv").set_index("Modèle")
    cost = read_table("resultats_cout_classifieur.csv", index_col=0)
    saving = cost.iloc[3, 2]
    c = st.columns(4)
    with c[0]:
        kpi("Mesures analysées", fr(len(ai4i)), "jeu AI4I 2020")
    with c[1]:
        kpi("Taux de pannes", f"{ai4i['Machine failure'].mean() * 100:.1f} %".replace(".", ","), "classes très déséquilibrées")
    with c[2]:
        kpi("ROC-AUC XGBoost", fr(clf.loc["XGBoost", "ROC-AUC"], 3), "classification binaire")
    with c[3]:
        kpi("Économie vs corrective", str(saving), "politique prédictive (étude de coût)")

    st.markdown("##### Ligne de production surveillée")
    st.markdown(
        '<div class="flow"><div class="step">Concassage</div><span class="arrow">›</span>'
        '<div class="step hl">Broyage cru</div><span class="arrow">›</span>'
        '<div class="step hl">Four rotatif · ventilateurs de tirage</div><span class="arrow">›</span>'
        '<div class="step hl">Refroidisseur clinker</div><span class="arrow">›</span>'
        '<div class="step hl">Broyage ciment</div><span class="arrow">›</span>'
        '<div class="step hl">Ensachage · compresseurs d\'air</div></div>', unsafe_allow_html=True)

    st.markdown("##### Trois stratégies de maintenance")
    c = st.columns(3)
    cards = [("Corrective", "On répare après la panne. Arrêts imprévus, ligne immobilisée, pièces à longs délais d'importation."),
             ("Préventive", "On remplace à âge fixe. Fiable, mais on change parfois des pièces encore bonnes."),
             ("Prédictive", "On surveille l'état réel de la machine et on intervient au bon moment : c'est l'objet de ce projet.")]
    for col, (t, d) in zip(c, cards):
        with col:
            cls = "card hl" if t == "Prédictive" else "card"
            st.markdown(f'<div class="{cls}"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    comp = read_table("comparaison_approches.csv")
    names = ["Corrective", "Préventive (âge fixe)", "Alarme anomalies", "Classifieur XGBoost", "Idéale (limite théorique)"]
    vals = comp["MAD/h"].round(1).tolist()
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[GRAY, GRAY, GRAY, PRIMARY, LINE],
                           text=[f"{fr(v)} MAD/h" for v in vals], textposition="outside"))
    fig.update_layout(title="Coût de maintenance par heure de fonctionnement (hypothèses illustratives)",
                      yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(vals) * 1.25]))
    show(fig, 340)

    with st.expander("Méthodologie et limites"):
        st.markdown(
            "- **Classification** : les modèles de cette interface (XGBoost, Random Forest, régression logistique) "
            "sont entraînés sur le jeu **AI4I 2020** (10 000 mesures de machines-outils).\n"
            "- **Parc simulé** : la supervision utilise des mesures du jeu de test, rebaptisées en équipements de cimenterie "
            "pour illustrer l'usage industriel.\n"
            "- **Survie et coûts** : calculés sur des données synthétiques (cycles de vie Weibull). Les coûts en dirhams "
            "sont des **hypothèses illustratives**.\n"
            "- Ces résultats valident la méthode ; ils ne mesurent pas la performance sur une usine réelle.")


# =========================================================================== PAGE 2
def page_fleet():
    header("SALLE DE CONTRÔLE", "Supervision du parc d'équipements",
           "État de santé de 12 équipements critiques, classés par niveau de risque. "
           "Sélectionnez un équipement pour ouvrir son diagnostic détaillé.")
    fleet = build_fleet().copy()
    fleet["p"] = [predict(fleet.iloc[[i]], MODEL_NAME) for i in range(len(fleet))]
    fleet = fleet.sort_values("p", ascending=False).reset_index(drop=True)
    status = [risk_status(p, THR) for p in fleet["p"]]
    n_alert = sum(s[0] == "Alerte" for s in status)
    n_watch = sum(s[0] == "Surveillance" for s in status)
    n_ok = sum(s[0] == "Normal" for s in status)

    c = st.columns(4)
    with c[0]:
        kpi("Équipements suivis", str(len(fleet)), "ventilateurs · broyeurs · compresseurs")
    with c[1]:
        kpi("En alerte", str(n_alert), "intervention requise", ALERT)
    with c[2]:
        kpi("Sous surveillance", str(n_watch), "risque à surveiller", WARN)
    with c[3]:
        kpi("Normaux", str(n_ok), "aucun écart notable", OK)

    st.markdown("")
    for start in range(0, len(fleet), 4):
        cols = st.columns(4)
        for col, i in zip(cols, range(start, min(start + 4, len(fleet)))):
            r, (label, color, _) = fleet.iloc[i], status[i]
            normal = label == "Normal"
            tone = GRAY if normal else color
            num = INK if normal else color
            pct = r.p * 100
            pct_txt = "< 1" if pct < 0.5 else f"{pct:.0f}"
            with col:
                st.markdown(
                    f'<div class="mc" style="--c:{tone};--n:{num}"><div class="mc-top"><span class="tag">{r.tag}</span>'
                    f'<span class="led"></span></div><div class="mc-name">{r["name"]}</div>'
                    f'<div class="mc-risk">{pct_txt}<small> % risque</small></div>'
                    f'<div class="bar"><div style="width:{max(r.p * 100, 2):.0f}%"></div></div>'
                    f'<div class="mc-meta">{r.rpm:.0f} tr/min · {r.torque:.0f} N·m · usure {r.tool_wear:.0f} min</div></div>',
                    unsafe_allow_html=True)

    st.markdown("##### Plan d'intervention")
    plan = []
    for i, r in fleet.iterrows():
        label = status[i][0]
        if label == "Normal":
            continue
        causes = [code for code, _, ratio, _, _ in physical_checks(r) if ratio >= 1]
        plan.append({"Priorité": "Urgente" if label == "Alerte" else "À planifier",
                     "Équipement": f"{r.tag} · {r['name']}", "Risque": float(r.p * 100),
                     "Modes de défaillance détectés": ", ".join(causes) if causes else "Dérive multi-capteurs",
                     "Action": status[i][2]})
    if plan:
        show_df(pd.DataFrame(plan), hide_index=True, column_config={
            "Risque": st.column_config.ProgressColumn("Risque (%)", min_value=0, max_value=100, format="%.0f %%")})
    else:
        st.success("Aucune intervention à planifier.")

    st.markdown("##### Détail d'un équipement")
    tag = st.selectbox("Équipement", fleet["tag"].tolist(), format_func=lambda t: f"{t} · {fleet.set_index('tag').loc[t, 'name']}")
    raw = fleet[fleet["tag"] == tag][FEATS].reset_index(drop=True)
    render_diagnostic(raw, MODEL_NAME, THR)


# =========================================================================== PAGE 3
PRESETS = {
    "Fonctionnement nominal": dict(g="M", air=25.0, proc=35.5, rpm=1500, torque=40.0, wear=50),
    "Surchauffe (ventilation dégradée)": dict(g="M", air=29.0, proc=36.5, rpm=1300, torque=45.0, wear=80),
    "Surcharge mécanique": dict(g="L", air=25.0, proc=35.0, rpm=1400, torque=62.0, wear=190),
    "Pièce d'usure en fin de vie": dict(g="M", air=25.0, proc=35.5, rpm=1500, torque=40.0, wear=235),
    "Anomalie de puissance": dict(g="M", air=25.0, proc=35.5, rpm=2500, torque=60.0, wear=50),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[f"d_{k}"] = v


def page_diagnostic():
    header("DIAGNOSTIC CONDITIONNEL", "Diagnostic d'un équipement",
           "Saisissez les relevés des capteurs ou chargez un scénario de défaillance : "
           "le modèle estime le risque de panne et identifie la cause probable.")
    if "d_g" not in st.session_state:
        apply_preset("Fonctionnement nominal")

    left, right = st.columns([1, 2.2])
    with left:
        st.markdown("**Scénarios types**")
        for name in PRESETS:
            st.button(name, on_click=apply_preset, args=(name,), key=f"btn_{name}")
        st.markdown("---")
        st.markdown("**Relevés capteurs**")
        g = st.radio("Classe de l'équipement", ["L", "M", "H"], horizontal=True, key="d_g")
        air = st.slider("Température ambiante (°C)", 20.0, 35.0, step=0.1, key="d_air")
        proc = st.slider("Température process (°C)", 30.0, 45.0, step=0.1, key="d_proc")
        rpm = st.slider("Vitesse de rotation (tr/min)", 1000, 3000, step=10, key="d_rpm")
        torque = st.slider("Couple (N·m)", 3.0, 80.0, step=0.5, key="d_torque")
        wear = st.slider("Usure de la pièce (min)", 0, 260, step=1, key="d_wear")
    with right:
        raw = build_row({"L": 0, "M": 1, "H": 2}[g], air, proc, rpm, torque, wear)
        render_diagnostic(raw, MODEL_NAME, THR)


# =========================================================================== PAGE 4
def page_reliability():
    header("FIABILITÉ · ANALYSE DE SURVIE", "Fiabilité et âge optimal de remplacement",
           "Modèle de Weibull ajusté sur les cycles de vie : probabilité de panne à court terme "
           "et âge de remplacement préventif qui minimise le coût.")
    surv = read_table("resultats_survie.csv")
    surv.columns = ["type", "rho", "ic", "beta", "lam", "eta", "tstar", "cout_opt", "cout_sans", "eco"]
    surv = surv.set_index("type")

    c1, c2, c3 = st.columns(3)
    with c1:
        eq = st.selectbox("Type d'équipement", list(surv.index), format_func=str.capitalize)
    row = surv.loc[eq]
    rho, lam, tstar = float(row.rho), float(row.lam), float(row.tstar)
    with c2:
        age = st.slider("Âge depuis la dernière maintenance (h)", 0, 4000, 600, 10)
    with c3:
        horizon = st.slider("Horizon de prévision (h)", 12, 240, 48, 12)

    rel = lambda t: np.exp(-((t / lam) ** rho))
    p_cond = 1 - rel(age + horizon) / rel(age)
    if age >= tstar:
        label, color, msg = "Remplacement recommandé", ALERT, "L'âge dépasse l'âge optimal de remplacement T*."
    elif age >= 0.8 * tstar:
        label, color, msg = "À planifier", WARN, "Prévoir le remplacement au prochain arrêt programmé."
    else:
        label, color, msg = "Exploitation normale", OK, "Poursuivre l'exploitation."

    k = st.columns(4)
    with k[0]:
        kpi("Fiabilité actuelle R(âge)", f"{rel(age) * 100:.0f} %", "probabilité d'être encore en service")
    with k[1]:
        kpi(f"Panne dans {horizon} h", f"{p_cond * 100:.1f} %".replace(".", ","), "sachant que l'équipement a survécu", color)
    with k[2]:
        kpi("Âge optimal T*", f"{fr(tstar)} h", "remplacement préventif")
    with k[3]:
        kpi("Économie du préventif", str(row.eco), "vs attendre la panne")
    st.markdown(f'<div style="margin:.8rem 0">{badge(label, color)} &nbsp; <span style="color:#9AA7B6">{msg}</span></div>',
               unsafe_allow_html=True)

    t = np.linspace(1, 5000, 400)
    haz = (rho / lam) * (t / lam) ** (rho - 1)
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=t, y=rel(t), line=dict(color=BLUE, width=3), name="R(t)"))
        fig.add_vline(x=age, line_color=ACCENT, annotation_text="âge actuel")
        fig.add_vline(x=tstar, line_color=OK, line_dash="dash", annotation_text="T*")
        fig.update_layout(title="Courbe de fiabilité R(t)", xaxis_title="heures", yaxis_title="probabilité de survie")
        show(fig, 340)
    with b:
        fig = go.Figure(go.Scatter(x=t, y=haz * 1000, line=dict(color=MUTED, width=3), name="h(t)"))
        fig.add_vline(x=age, line_color=ACCENT, annotation_text="âge actuel")
        fig.add_vline(x=tstar, line_color=OK, line_dash="dash", annotation_text="T*")
        fig.update_layout(title="Taux de défaillance h(t) : l'usure fait croître le risque",
                          xaxis_title="heures", yaxis_title="pannes par 1 000 h")
        show(fig, 340)

    st.markdown("##### Paramètres estimés par type d'équipement")
    table = surv[["rho", "lam", "tstar", "eco"]].rename(columns={
        "rho": "Forme ρ (usure si > 1)", "lam": "Échelle λ (h)", "tstar": "T* (h)", "eco": "Économie"})
    show_df(table.round(2))
    with st.expander("Courbes de Kaplan-Meier et Weibull (notebook 05)"):
        a, b = st.columns(2)
        with a:
            show_image("survie_kaplan_meier.png")
        with b:
            show_image("survie_weibull_vs_km.png")


# =========================================================================== PAGE 5
def page_economics():
    header("ANALYSE ÉCONOMIQUE", "Combien rapporte la maintenance prédictive ?",
           "Coût de chaque politique de maintenance en dirhams (MAD). "
           "Les coûts unitaires sont des hypothèses illustratives à remplacer par des chiffres justifiés.")
    cost = read_table("resultats_cout_classifieur.csv", index_col=0)
    cost.columns = ["mad_h", "mad_an", "eco"]
    names = list(cost.index)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_mach = st.slider("Équipements surveillés", 1, 200, 20)
    with c2:
        ref = st.selectbox("Politique de référence", names[:2], index=0)
    with c3:
        strat = st.selectbox("Politique à évaluer", names[2:], index=2)
    hours = 8000
    gain_machine = (cost.loc[ref, "mad_h"] - cost.loc[strat, "mad_h"]) * hours
    gain_total = gain_machine * n_mach

    k = st.columns(3)
    with k[0]:
        kpi("Gain annuel / équipement", f"{fr(gain_machine / 1e6, 2)} M MAD", f"{hours} h de fonctionnement par an")
    with k[1]:
        kpi(f"Gain annuel · {n_mach} équipements", f"{fr(gain_total / 1e6, 1)} M MAD", "vs politique de référence")
    with k[2]:
        kpi("Économie relative", f"{(1 - cost.loc[strat, 'mad_h'] / cost.loc[ref, 'mad_h']) * 100:.0f} %", "du coût horaire")

    st.markdown("")
    a, b = st.columns([1.1, 1])
    with a:
        colors = [GRAY, GRAY, GRAY, PRIMARY, PRIMARY, LINE][: len(names)]
        fig = go.Figure(go.Bar(x=cost["mad_an"] / 1e6, y=names, orientation="h", marker_color=colors,
                               text=[f"{fr(v / 1e6, 2)} M" for v in cost["mad_an"]], textposition="outside"))
        fig.update_layout(title="Coût annuel par équipement (M MAD)", yaxis=dict(autorange="reversed"),
                          xaxis=dict(range=[0, cost["mad_an"].max() / 1e6 * 1.2]))
        show(fig, 380)
    with b:
        sens = read_table("resultats_cout_sensibilite.csv")
        sens.columns = ["facteur", "corrective", "fixe", "alarme", "idéale"]
        fig = go.Figure()
        for col, nm, colr, dash, w in [("corrective", "Corrective", MUTED, "solid", 2), ("fixe", "Âge fixe", GRAY, "dash", 2),
                                       ("alarme", "Alarme prédictive", PRIMARY, "solid", 3), ("idéale", "Idéale", OK, "dot", 2)]:
            fig.add_trace(go.Scatter(x=sens["facteur"], y=sens[col], name=nm, mode="lines+markers",
                                     line=dict(color=colr, dash=dash, width=w)))
        fig.update_layout(title="Sensibilité au coût de l'arrêt de production",
                          xaxis=dict(type="log", title="facteur appliqué à la perte de production"), yaxis_title="MAD / h")
        show(fig, 380)
    st.caption("Plus l'arrêt de production coûte cher, plus l'écart entre maintenance corrective et prédictive se creuse.")


# =========================================================================== PAGE 6
def page_models():
    header("PERFORMANCES · EXPLICABILITÉ", "Modèles et explicabilité",
           "Comparaison des classifieurs, des détecteurs d'anomalies et des variables qui pilotent les alertes (SHAP).")
    t1, t2, t3, t4 = st.tabs(["Classification AI4I", "Cimenterie simulée", "Détection d'anomalies", "Explicabilité SHAP"])

    with t1:
        df = read_table("resultats_classification.csv")
        metrics = ["PR-AUC", "ROC-AUC", "Recall", "Précision", "F1"]
        fig = go.Figure()
        for m, c in zip(metrics, [PRIMARY, "#4F7CAC", "#8FAAC8", "#C3CFDC", "#7B8794"]):
            fig.add_trace(go.Bar(x=df["Modèle"], y=df[m], name=m, marker_color=c))
        fig.update_layout(barmode="group", title="Métriques sur le jeu de test AI4I", yaxis=dict(range=[0, 1.05]))
        show(fig, 380)
        show_df(df, hide_index=True)
        a, b = st.columns(2)
        with a:
            show_image("confusion_matrices.png")
        with b:
            show_image("pr_curves.png")
    with t2:
        df = read_table("resultats_classification_synthetique.csv")
        show_df(df, hide_index=True)
        a, b = st.columns(2)
        with a:
            show_image("classif_synth_pr_curves.png")
        with b:
            show_image("classif_synth_confusion.png")
    with t3:
        df = read_table("resultats_anomalies.csv")
        show_df(df, hide_index=True)
        st.caption("Le détecteur One-Class SVM trouve 43 pannes sur 44 sans utiliser d'étiquettes de panne.")
        a, b = st.columns(2)
        with a:
            show_image("anomalies_pr_curves.png")
        with b:
            show_image("anomalies_exemple_machine.png")
    with t4:
        shap = read_table("shap_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)[:15][::-1]
        fig = go.Figure(go.Bar(x=shap.values, y=[pretty_feature(i) for i in shap.index], orientation="h", marker_color=ACCENT))
        fig.update_layout(title="Variables les plus influentes (|SHAP| moyen)", xaxis_title="impact moyen sur la prédiction")
        show(fig, 520)
        st.caption("Les vibrations (RMS, kurtosis) et leur dérive par rapport à l'état sain dominent : "
                   "c'est cohérent avec la pratique de la surveillance vibratoire des machines tournantes.")
        a, b = st.columns(2)
        with a:
            show_image("shap_summary.png")
        with b:
            show_image("shap_exemple_alerte.png")


# --------------------------------------------------------------------------- routage
{
    PAGES[0]: page_overview, PAGES[1]: page_fleet, PAGES[2]: page_diagnostic,
    PAGES[3]: page_reliability, PAGES[4]: page_economics, PAGES[5]: page_models,
}[page]()
