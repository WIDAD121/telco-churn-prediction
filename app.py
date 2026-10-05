"""Streamlit app: churn risk for one customer or for a CSV file of customers.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from churn.data import RAW_FEATURES
from churn.predict import explain, load_artifacts, make_customer, predict_proba

st.set_page_config(page_title="Prédiction du churn Telco", page_icon="📡", layout="wide")

YES_NO = {"No": "Non", "Yes": "Oui"}
LABELS = {
    "gender": "Genre", "SeniorCitizen": "Senior (65 ans et plus)", "Partner": "En couple",
    "Dependents": "Personnes à charge", "tenure": "Ancienneté", "PhoneService": "Téléphonie",
    "MultipleLines": "Lignes multiples", "InternetService": "Internet",
    "OnlineSecurity": "Sécurité en ligne", "OnlineBackup": "Sauvegarde en ligne",
    "DeviceProtection": "Protection de l'appareil", "TechSupport": "Support technique",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming films", "Contract": "Contrat",
    "PaperlessBilling": "Facture dématérialisée", "PaymentMethod": "Mode de paiement",
    "MonthlyCharges": "Charges mensuelles", "TotalCharges": "Charges totales",
    "n_addons": "Nombre d'options Internet",
}
CHOICES = {
    "gender": {"Female": "Femme", "Male": "Homme"},
    "InternetService": {"DSL": "DSL", "Fiber optic": "Fibre optique", "No": "Aucun"},
    "Contract": {"Month-to-month": "Mensuel", "One year": "Un an", "Two year": "Deux ans"},
    "PaymentMethod": {
        "Electronic check": "Chèque électronique", "Mailed check": "Chèque postal",
        "Bank transfer (automatic)": "Virement automatique",
        "Credit card (automatic)": "Carte bancaire automatique",
    },
}


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def choice(column: str, default: str = None, disabled: bool = False) -> str:
    """Select box that shows French labels and returns the dataset value."""
    options = CHOICES.get(column, YES_NO)
    keys = list(options)
    index = keys.index(default) if default in keys else 0
    return st.selectbox(LABELS[column], keys, index=index, disabled=disabled,
                        format_func=lambda k: options[k], key=column)


try:
    pipeline, meta = get_artifacts()
except Exception as error:  # model missing or trained with another library version
    st.error("Modèle introuvable ou incompatible. Lancez d'abord `python -m churn.train` "
             f"à la racine du projet.\n\nDétail : {error}")
    st.stop()

st.title("📡 Prédiction du churn client")
st.caption("Estime la probabilité qu'un client résilie son abonnement télécom.")

with st.sidebar:
    st.header("Paramètres")
    threshold = st.slider("Seuil de décision", 0.10, 0.90, float(meta["threshold"]), 0.01)
    st.caption(
        f"Seuil recommandé : {meta['threshold']:.2f}. Sur le jeu de test, il détecte "
        f"{meta['test_recall_at_threshold']:.0%} des churners, et "
        f"{meta['test_precision_at_threshold']:.0%} des alertes sont de vrais churners. "
        "Baissez-le pour en détecter plus, au prix de plus de fausses alertes."
    )
    st.divider()
    st.markdown(f"**Modèle** : {meta['model']}")
    st.markdown(f"**ROC-AUC (test)** : {meta['test_roc_auc']:.3f}")
    st.markdown(f"**PR-AUC (test)** : {meta['test_pr_auc']:.3f}")

tab_one, tab_file = st.tabs(["Un client", "Fichier CSV"])

# ------------------------------------------------------------- one customer --
with tab_one:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil")
        gender = choice("gender")
        senior = choice("SeniorCitizen")
        partner = choice("Partner")
        dependents = choice("Dependents")
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone = choice("PhoneService", default="Yes")
        multiple = choice("MultipleLines", disabled=phone == "No")
        internet = choice("InternetService", default="Fiber optic")
        no_internet = internet == "No"
        security = choice("OnlineSecurity", disabled=no_internet)
        backup = choice("OnlineBackup", disabled=no_internet)
        protection = choice("DeviceProtection", disabled=no_internet)
        support = choice("TechSupport", disabled=no_internet)
        tv = choice("StreamingTV", disabled=no_internet)
        movies = choice("StreamingMovies", disabled=no_internet)

    with col3:
        st.subheader("Contrat et facturation")
        contract = choice("Contract")
        paperless = choice("PaperlessBilling", default="Yes")
        payment = choice("PaymentMethod")
        monthly = st.number_input("Charges mensuelles ($)", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input("Charges totales ($)", 0.0, 10000.0, float(monthly * tenure), step=1.0,
                                help="Par défaut : charges mensuelles x ancienneté.")

    if st.button("Prédire", type="primary"):
        customer = make_customer(
            gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
            tenure=tenure, PhoneService=phone, MultipleLines=multiple, InternetService=internet,
            OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=protection,
            TechSupport=support, StreamingTV=tv, StreamingMovies=movies, Contract=contract,
            PaperlessBilling=paperless, PaymentMethod=payment,
            MonthlyCharges=monthly, TotalCharges=total,
        )
        proba = float(predict_proba(pipeline, customer).iloc[0])
        at_risk = proba >= threshold

        left, right = st.columns(2)
        left.metric("Probabilité de churn", f"{proba:.1%}")
        right.metric("Décision", "Client à risque" if at_risk else "Client fidèle")
        st.progress(min(max(proba, 0.0), 1.0))
        if at_risk:
            st.warning(f"Probabilité au-dessus du seuil ({threshold:.0%}) : une action de rétention est conseillée.")
        else:
            st.success(f"Probabilité en dessous du seuil ({threshold:.0%}) : pas d'action nécessaire.")

        contributions = explain(pipeline, customer)
        if contributions is not None:
            st.subheader("Ce qui explique ce score")
            top = contributions.head(6).rename(index=lambda name: LABELS.get(name, name))
            st.bar_chart(top.rename("Contribution au score"), horizontal=True)
            st.caption("Valeur positive : le facteur augmente le risque de churn. "
                       "Valeur négative : il le réduit. Les contributions sont exprimées en log-odds.")

# ------------------------------------------------------------------ CSV file --
with tab_file:
    st.write("Chargez un fichier CSV au format du jeu de données Telco pour scorer plusieurs clients.")
    with st.expander("Colonnes attendues"):
        st.code(", ".join(RAW_FEATURES))
    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is not None:
        try:
            table = pd.read_csv(uploaded)
            scored = table.copy()
            scored["churn_probability"] = predict_proba(pipeline, table).round(4)
            scored["at_risk"] = scored["churn_probability"] >= threshold
            scored = scored.sort_values("churn_probability", ascending=False)
            st.metric("Clients à risque", f"{int(scored['at_risk'].sum())} sur {len(scored)}")
            st.dataframe(scored)
            st.download_button("Télécharger les scores", scored.to_csv(index=False).encode("utf-8"),
                               file_name="churn_scores.csv", mime="text/csv")
        except Exception as error:
            st.error(f"Lecture ou scoring impossible : {error}")
