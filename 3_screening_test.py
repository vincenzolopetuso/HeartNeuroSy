# 3_screening_test.py
# ------------------------------------------------------------
# OBIETTIVO
#   Valutare lo scenario "screening" (low-data):
#   confrontiamo un modello che usa SOLO feature a basso costo (cheap)
#   vs lo stesso modello con l'aggiunta delle feature semantiche inferite dalla KB.
#
# IDEA SPERIMENTALE
#   Nel setting screening si dispone di pochi esami/variabili.
#   La KB può aggiungere background knowledge (pattern discreti) e aiutare
#   leggermente la predizione, anche quando i dati sono limitati.
#
# INPUT
#   - heart_dataset_enriched.csv
#
# OUTPUT
#   - stampa a console di media ± std (CV ripetuta) per più metriche
#   - file results_cache.txt in formato CSV (scenario, metric, mean, std)
#     usato dallo script 4 per produrre i grafici.
# ------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


CACHE_FILE = "results_cache.txt"


def summarize(scores: dict, metric: str):
    """Calcola media e deviazione standard dei punteggi di test per una metrica."""
    vals = scores[f"test_{metric}"]
    return float(np.mean(vals)), float(np.std(vals))


def run_eval(name: str, model, X, y, cv):
    """
    Esegue cross-validate con più metriche e ritorna un dizionario:
      metric -> (mean, std)
    """
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
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

    print(f"\n🧠 {name}")
    out = {}
    for m in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        mean, std = summarize(scores, m)
        out[m] = (mean, std)
        print(f"  - {m:16s}: {mean:.4f} ± {std:.4f}")
    return out


def run_screening_test():
    print("\n🏥 SCENARIO SCREENING (Low-Data)...")

    df = pd.read_csv("data/heart_dataset_enriched.csv")

    # Target binario
    if "target" not in df.columns:
        raise ValueError("Colonna 'target' non trovata. Assicurati di aver eseguito build_kb.py.")

    y = df["target"].astype(int)

    # -----------------------------
    # Feature "cheap" (a basso costo)
    # -----------------------------
    # Scelta volutamente ridotta: simula uno screening con pochi esami facilmente disponibili.
    cheap_features = [
        "age", "sex", "cp", "trestbps", "fbs", "restecg", "thalach", "exang"
    ]

    missing_cheap = [c for c in cheap_features if c not in df.columns]
    if missing_cheap:
        raise ValueError(f"Mancano queste feature cheap nel CSV: {missing_cheap}")

    X_cheap = df[cheap_features]

    # -----------------------------
    # Feature semantiche (auto-detect)
    # -----------------------------
    # Manteniamo i nomi coerenti con build_kb.py.
    expected_semantic = [
        "is_Hypertensive",
        "is_SevereHypertensive",
        "is_Hyperchol",
        "is_HighSugar",
        "is_Older",
        "is_MetabolicSyndromeSuspect",
        "is_SilentHighRisk",
    ]
    semantic_features = [c for c in expected_semantic if c in df.columns]

    print("✅ Feature semantiche trovate:", semantic_features)
    missing_sem = [c for c in expected_semantic if c not in df.columns]
    if missing_sem:
        print("⚠️  Feature semantiche mancanti (non è un errore):", missing_sem)

    if len(semantic_features) == 0:
        print("❌ Nessuna feature semantica trovata: esegui build_kb.py e verifica l'output.")
        return

    # Dataset screening arricchito con KB
    X_cheap_kb = df[cheap_features + semantic_features]

    # -----------------------------
    # Modello: Logistic Regression + scaling
    # -----------------------------
    # Scelta motivata: modello semplice, interpretabile e stabile su dataset piccoli.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])

    # CV stratificata ripetuta: risultati più robusti (media ± std)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

    # Esecuzione confronto
    res_cheap = run_eval("LogReg (SOLO cheap)", model, X_cheap, y, cv)
    res_kb = run_eval("LogReg (cheap + KB)", model, X_cheap_kb, y, cv)

    # -----------------------------
    # Cache risultati per plotting
    # -----------------------------
    # Salviamo mean e std per scenario e metrica in formato CSV.
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write("scenario,metric,mean,std\n")
        for metric in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
            m1, s1 = res_cheap[metric]
            m2, s2 = res_kb[metric]
            f.write(f"cheap,{metric},{m1},{s1}\n")
            f.write(f"cheap_kb,{metric},{m2},{s2}\n")

    print(f"\n💾 Cache salvata in: {CACHE_FILE}")
    print("✅ Screening test completato.")


if __name__ == "__main__":
    run_screening_test()

