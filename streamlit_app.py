import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rakuten MLOps", page_icon="🛍️")

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📋 Contexte", "📊 Exploration", "🔮 Prédiction"])

# ============================================================
# ONGLET 1 — Contexte
# ============================================================
with tab1:
    st.title("🛍️ Rakuten — Classification Multimodale")
    st.markdown("""
    ## Description du projet
    Prédire le **code type** d'un produit Rakuten à partir de sa désignation et description textuelle.
    
    ## Stack technique
    - **Preprocessing** : BeautifulSoup, NLTK, spaCy
    - **Modèle** : TF-IDF + Régression Logistique
    - **API** : FastAPI
    - **Tracking** : MLflow
    - **Conteneurisation** : Docker
    """)
    
    st.metric("Weighted F1 baseline", "0.8056")
    st.metric("Benchmark Rakuten", "0.8100")

# ============================================================
# ONGLET 2 — Exploration
# ============================================================
with tab2:
    st.title("📊 Exploration des données")
    
    df_Y = pd.read_csv("Y_train_CVw08PX.csv", index_col=0)
    
    st.subheader("Distribution des classes")
    fig, ax = plt.subplots(figsize=(12, 5))
    df_Y["prdtypecode"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Code produit")
    ax.set_ylabel("Nombre de produits")
    st.pyplot(fig)
    
    st.subheader("Statistiques")
    st.write(f"Nombre de produits : **{len(df_Y)}**")
    st.write(f"Nombre de classes : **{df_Y['prdtypecode'].nunique()}**")

# ============================================================
# ONGLET 3 — Prédiction
# ============================================================
with tab3:
    st.title("🔮 Prédiction")
    
    text_input = st.text_area("Décris ton produit :", 
                               placeholder="Ex: Jeux Nintendo Switch Mario Kart console")
    
    if st.button("Prédire"):
        if text_input:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text_input}
            )
            if response.status_code == 200:
                categorie = response.json()["categorie"]
                st.success(f"✅ Catégorie prédite : **{categorie}**")
            else:
                st.error("❌ Erreur — l'API ne répond pas")
        else:
            st.warning("⚠️ Entre un texte d'abord !")