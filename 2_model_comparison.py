# 2_model_comparison.py
# ------------------------------------------------------------
# OBIETTIVO
#   Confrontare prestazioni di modelli ML in due setting:
#     (A) BASELINE  : solo feature originali del dataset
#     (B) ENRICHED  : feature originali + feature semantiche inferite dalla KB
#
# VALUTAZIONE "SCIENTIFICA"
#   - Cross-validation stratificata ripetuta (10-fold x 5 repeats)
#   - Metriche multiple riportate come media ± deviazione standard:
#       accuracy, balanced_accuracy, f1, roc_auc
#
# INPUT
#   - heart_dataset_enriched.csv (generato da build_kb.py)
#
# OUTPUT
#   - stampa a console dei risultati (media ± std) per ogni modello e setting
# ------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def summarize_scores(scores: dict, metric_key: str):
    """
    Utility: data la struttura restituita da cross_validate,
    calcola media e deviazione standard per la metrica richiesta.
    """
    vals = scores[f"test_{metric_key}"]
    return float(np.mean(vals)), float(np.std(vals))


def run_model(name: str, model, X, y, cv):
    """
    Esegue cross-validation con più metriche e stampa un report sintetico.
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    print(f"\n🧠 Modello: {name}")
    for m in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        mean, std = summarize_scores(scores, m)
        print(f"  - {m:16s}: {mean:.4f} ± {std:.4f}")

    return scores


def run_comparison():
    print("\n📊 VALUTAZIONE SCIENTIFICA (Metriche Multiple)...")

    # Dataset arricchito (contiene anche le colonne is_*)
    df = pd.read_csv("data/heart_dataset_enriched.csv")

    # Target binario (0/1)
    if "target" not in df.columns:
        raise ValueError("Colonna 'target' non trovata nel CSV. Assicurati di aver eseguito build_kb.py.")

    y = df["target"].astype(int)

    # Feature numeriche/categoriche originali (baseline)
    feat_numeric = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]
    missing_num = [c for c in feat_numeric if c not in df.columns]
    if missing_num:
        raise ValueError(f"Mancano queste feature numeriche nel CSV: {missing_num}")

    X_base = df[feat_numeric]

    # Feature semantiche attese (nomi coerenti con build_kb.py)
    # Nota: usiamo auto-detect per robustezza (non tutti i CSV potrebbero averle tutte).
    expected_semantic = [
        "is_Hypertensive",
        "is_SevereHypertensive",
        "is_Hyperchol",
        "is_HighSugar",
        "is_Older",
        "is_MetabolicSyndromeSuspect",
        "is_SilentHighRisk",
    ]

    feat_semantic = [c for c in expected_semantic if c in df.columns]
    print("✅ Feature semantiche trovate:", feat_semantic)

    missing_sem = [c for c in expected_semantic if c not in df.columns]
    if missing_sem:
        print("⚠️  Feature semantiche mancanti nel CSV (non è un errore):", missing_sem)

    # Se mancano tutte le feature semantiche, il confronto enriched non ha senso
    if len(feat_semantic) == 0:
        print("❌ Nessuna feature semantica trovata: il confronto Enriched non avrebbe senso.")
        print("   Controlla che build_kb.py abbia generato colonne is_* e che reasoning sia andato a buon fine.")
        return

    X_enriched = df[feat_numeric + feat_semantic]

    # CV stratificata ripetuta: riduce varianza e rende i risultati più stabili
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

    # Modello 1: Logistic Regression (con scaling)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])

    # Modello 2: Random Forest (non richiede scaling, robusto su feature miste)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    # Esecuzione: baseline
    print("\n==================== BASELINE ====================")
    run_model("LogReg (Baseline)", lr, X_base, y, cv)
    run_model("RandomForest (Baseline)", rf, X_base, y, cv)

    # Esecuzione: enriched (+KB)
    print("\n==================== ENRICHED (+KB) ====================")
    run_model("LogReg (Enriched)", lr, X_enriched, y, cv)
    run_model("RandomForest (Enriched)", rf, X_enriched, y, cv)

    print("\n✅ Valutazione completata.")


if __name__ == "__main__":
    run_comparison()
