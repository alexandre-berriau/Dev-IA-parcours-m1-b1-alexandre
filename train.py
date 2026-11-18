"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Reprend les steps des notebooks 01 à 04 :
- extraction SQL sans les colonnes qui trahissent la réponse (notebook 02) ;
- découpage temporel, les 4 derniers mois d'arrivée servent de test (notebook 02) ;
- pipeline sur données brutes avec 4 variables créées (notebook 03) ;
- forêt aléatoire réglée et seuil de décision à 0,30 (notebooks 03 et 04).

"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4
TARGET = "is_canceled"

# seuil de décision choisi sur les 4 derniers mois du jeu d'entraînement
THRESHOLD = 0.30

load_dotenv(ROOT / ".env")

# colonnes texte
CATEGORICAL = [
    "hotel", "arrival_date_month", "market_segment", "deposit_type",
    "customer_type", "country", "meal", "distribution_channel",
    "reserved_room_type",
]

# colonnes numériques
NUMERICAL = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "adr", "previous_cancellations",
    "previous_bookings_not_canceled", "total_of_special_requests",
    "required_car_parking_spaces", "is_repeated_guest", "days_in_waiting_list",
    "duree_sejour", "taille_groupe", "avec_enfants", "taux_annulation_passe",
]


def load_data(engine) -> pd.DataFrame:
    """
    Charge les réservations et le code de leur hôtel, en une requête.
    """
    query = text("""
        SELECT
            r.is_canceled,
            r.arrival_date,
            h.code AS hotel,
            r.arrival_date_month,
            r.market_segment,
            r.deposit_type,
            r.customer_type,
            r.country,
            r.meal,
            r.distribution_channel,
            r.reserved_room_type,
            r.lead_time,
            r.stays_in_weekend_nights,
            r.stays_in_week_nights,
            r.adults,
            r.children,
            r.babies,
            r.adr,
            r.previous_cancellations,
            r.previous_bookings_not_canceled,
            r.total_of_special_requests,
            r.required_car_parking_spaces,
            r.is_repeated_guest,
            r.days_in_waiting_list
        FROM reservations r
        JOIN hotels h ON r.hotel_id = h.hotel_id
    """)
    return pd.read_sql(query, engine, parse_dates=["arrival_date"])


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Met de côté les 4 derniers mois d'arrivée comme jeu de test.
    """
    bascule = (df["arrival_date"].max().to_period("M") - (TEST_MONTHS - 1)).to_timestamp()
    train = df[df["arrival_date"] < bascule].copy()
    test = df[df["arrival_date"] >= bascule].copy()
    return train, test


def ajouter_variables(donnees: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les 4 variables créées, puis ne garde que les colonnes du modèle.
    """
    d = donnees.copy()

    enfants = d["children"].fillna(0) + d["babies"]  # children a des valeurs manquantes

    d["duree_sejour"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["taille_groupe"] = d["adults"] + enfants
    d["avec_enfants"] = (enfants > 0).astype(int)

    historique = d["previous_cancellations"] + d["previous_bookings_not_canceled"]
    # part des annulations dans le passé du client, 0 si on ne le connaît pas
    d["taux_annulation_passe"] = np.where(
        historique > 0,
        d["previous_cancellations"] / historique.replace(0, np.nan), 
        0.0,
    )

    return d[CATEGORICAL + NUMERICAL]


def build_pipeline() -> Pipeline:
    """Pipeline complet : données brutes en entrée, prédiction en sortie."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                # min_frequency=50 : les modalités rares sont regroupées, country en compte 148
                ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
            ]), CATEGORICAL),
            ("num", SimpleImputer(strategy="median"), NUMERICAL),  # médiane : robuste aux extrêmes
        ],
        sparse_threshold=0,  # sortie dense, les forêts n'aiment pas les matrices creuses
    )

    classifier = RandomForestClassifier(
        n_estimators=200,      # nombre d'arbres dans la forêt
        min_samples_leaf=100,  # une feuille couvre au moins 100 réservations
        max_features=0.7,      # chaque arbre voit 70 % des colonnes
        random_state=42,       # pour la reproductibilité
        n_jobs=-1,             # utilise tous les cœurs disponibles (CPU uniquement et pas GPU)
    )

    return Pipeline([
        ("variables", FunctionTransformer(ajouter_variables)),
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Mesure le modèle au seuil THRESHOLD."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": THRESHOLD,
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),  # indépendant du seuil
        "confusion_matrix": {
            "vrais_negatifs": int(tn),
            "fausses_alertes": int(fp),
            "annulations_ratees": int(fn),
            "vrais_positifs": int(tp),
        },
    }


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    model_path = MODEL_DIR / "cancellation_model_v2.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": "2.0.0",
        "trained_on": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "target": TARGET,
        "train_period": [str(train["arrival_date"].min()), str(train["arrival_date"].max())],
        "test_period": [str(test["arrival_date"].min()), str(test["arrival_date"].max())],
        "hyperparameters": {
            k: v for k, v in model.get_params().items()
            if k.startswith("classifier__") and isinstance(v, (int, float, str, bool, type(None)))
        },
        "test_metrics": metrics,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (MODEL_DIR / "model_info_v2.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    import train

    train.main()
