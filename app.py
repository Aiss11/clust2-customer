"""
Streamlit — Segmentation des clients télécom (clustering DBSCAN, Customer_dataset)
Reprend l'application de clustering du prof : on recharge les points cœur de DBSCAN
sauvegardés dans le notebook et on applique sa règle à un nouveau client.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Segmentation clients", page_icon="📡", layout="centered")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    return jb.load("modele_dbscan.joblib")


artefacts = charger_objets()
points_coeur = artefacts["points_coeur"]
labels_coeur = artefacts["labels_coeur"]
eps = artefacts["eps"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
exemples = artefacts["exemples"]
noms_classes = artefacts["noms_classes"]


# ---------- Règle de DBSCAN pour un nouveau client ----------
def predire_classe(nouveau_client):
    client = normalize(np.array(nouveau_client, dtype=float).reshape(1, -1))
    distances = np.linalg.norm(points_coeur - client, axis=1)
    plus_proche = distances.argmin()
    if distances[plus_proche] <= eps:
        return int(labels_coeur[plus_proche])
    return -1


def libelle(classe):
    return "Client atypique (anomalie)" if classe == -1 else noms_classes[classe]


def predire_fichier(fichier):
    tableau = pd.read_csv(fichier)
    tableau["Segment"] = [libelle(predire_classe(ligne[:len(colonnes)])) for ligne in tableau.values]
    return tableau


st.title("📡 Segmentation des clients")
st.caption("À quel segment appartient ce client ? Modèle DBSCAN entraîné sur 3 150 clients d'un opérateur télécom.")
onglet_un, onglet_csv = st.tabs(["Un client", "Fichier CSV"])

with onglet_un:
    depart = st.selectbox("Pré-remplir avec", ["Valeurs médianes", "Client 1", "Client 2", "Client 3"])
    defauts = valeurs_defaut if depart == "Valeurs médianes" else exemples[int(depart[-1]) - 1]
    valeurs = []
    gauche, droite = st.columns(2)
    for i, (nom, val) in enumerate(zip(colonnes, defauts)):
        zone = gauche if i % 2 == 0 else droite
        valeurs.append(zone.number_input(nom.replace("  ", " "), value=float(val), key=f"{nom}_{depart}"))

    if st.button("Trouver le segment", type="primary", use_container_width=True):
        classe = predire_classe(valeurs)
        if classe == -1:
            st.warning("**Client atypique (anomalie)** : il ne ressemble à aucun segment.")
        else:
            st.success(f"**Segment :** {noms_classes[classe]}")

with onglet_csv:
    st.info("Le fichier doit contenir les 12 colonnes du jeu de données, dans le même ordre.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = predire_fichier(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Télécharger les résultats", tableau.to_csv(index=False).encode("utf-8"),
                               "segments_clients.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
