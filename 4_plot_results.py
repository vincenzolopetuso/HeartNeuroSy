# 4_plot_results.py
# ------------------------------------------------------------
# OBIETTIVO
#   Generare 4 grafici separati (uno per metrica) a partire dai risultati
#   salvati in results_cache.txt (prodotto da 3_screening_test.py).
#
# NOTA
#   I grafici mostrano:
#     - media (%) della metrica
#     - barre d'errore = ± deviazione standard (CV ripetuta)
#   Si usa uno "zoom" sull'asse Y per rendere visibili differenze piccole.
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# File cache generato dallo script 3 (formato CSV con header)
CACHE_FILE = "results_cache.txt"

# Su Windows: se True, apre automaticamente le immagini generate
AUTO_OPEN = True

# Etichette e colori coerenti con la narrazione del progetto
LABELS = ["ML Standard\n(Dati Limitati)", "Neuro-Symbolic AI\n(Dati + Ontologia)"]
COLORS = ["#95a5a6", "#2ecc71"]  # Grigio (baseline) e verde (KB)

# Titoli leggibili (in percentuale)
METRIC_TITLES = {
    "accuracy": "Accuratezza (%)",
    "balanced_accuracy": "Balanced Accuracy (%)",
    "f1": "F1-Score (%)",
    "roc_auc": "ROC-AUC (%)",
}


def plot_metric(df, metric_name):
    """
    Genera e salva il grafico per una specifica metrica.
    df: dataframe con colonne (scenario, metric, mean, std)
    metric_name: string (es. "accuracy")
    """
    # 1) Filtriamo il dataframe per la metrica corrente
    subset = df[df['metric'] == metric_name]

    if subset.empty:
        print(f"[WARN] Nessun dato trovato per la metrica: {metric_name}")
        return None

    # 2) Estraiamo i dati per i due scenari attesi (cheap e cheap_kb)
    # cheap     = baseline screening (solo feature low-cost)
    # cheap_kb  = screening + feature semantiche inferite dalla KB
    try:
        row_base = subset[subset['scenario'] == 'cheap'].iloc[0]
        row_kb = subset[subset['scenario'] == 'cheap_kb'].iloc[0]
    except IndexError:
        print(f"[ERR] Dati incompleti per {metric_name}. Controlla results_cache.txt")
        return None

    # Convertiamo in percentuale (0-100) per il grafico
    means = [row_base['mean'] * 100, row_kb['mean'] * 100]
    stds = [row_base['std'] * 100, row_kb['std'] * 100]

    # 3) Creazione grafico
    plt.figure(figsize=(8, 6))

    # Barre con error bars (± deviazione standard)
    bars = plt.bar(
        LABELS,
        means,
        yerr=stds,
        capsize=10,
        color=COLORS,
        edgecolor='black',
        alpha=0.9,
        width=0.6
    )

    # Titoli e assi
    title_metric = METRIC_TITLES.get(metric_name, metric_name)
    plt.title(f"Confronto: {title_metric}", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(title_metric, fontsize=11)

    # Zoom intelligente sull'asse Y: consideriamo anche le barre d'errore
    min_y = min([m - s for m, s in zip(means, stds)])
    max_y = max([m + s for m, s in zip(means, stds)])
    plt.ylim(max(0, min_y - 5), min(100, max_y + 5))

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 4) Etichette sopra le barre: mostrano la media in percentuale
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + (max_y - min_y) * 0.05,
            f'{height:.2f}%',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    # Nota in basso: esplicita il significato delle barre d'errore
    plt.text(
        0.5, -0.15,
        "Barre d'errore: ± Deviazione Standard (CV Ripetuta)",
        ha='center', transform=plt.gca().transAxes,
        fontsize=9, style='italic'
    )

    # 5) Salvataggio
    filename = f"risultati_screening_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"✅ Grafico salvato: {filename}")
    return filename


def main():
    print("🎨 Generazione Grafici dei Risultati...")

    # 1) Controllo esistenza cache
    if not os.path.exists(CACHE_FILE):
        print(f"❌ ERRORE: Il file '{CACHE_FILE}' non esiste.")
        print("   -> Esegui prima lo script '3_screening_test.py' per generare i dati.")
        return

    # 2) Caricamento dati (CSV)
    try:
        df = pd.read_csv(CACHE_FILE)
    except Exception as e:
        print(f"❌ ERRORE nella lettura del CSV: {e}")
        return

    # Verifica formato minimo
    required_cols = {'scenario', 'metric', 'mean', 'std'}
    if not required_cols.issubset(df.columns):
        print(f"❌ ERRORE: Formato CSV non valido. Colonne attese: {required_cols}")
        return

    # 3) Generazione grafici per le metriche di interesse
    metrics_to_plot = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    generated_files = []

    for metric in metrics_to_plot:
        out_file = plot_metric(df, metric)
        if out_file:
            generated_files.append(out_file)

    print("\n" + "="*50)
    print("🎉 TUTTI I GRAFICI GENERATI CORRETTAMENTE!")
    print("="*50)

    # 4) Apertura automatica (solo su Windows)
    if AUTO_OPEN and os.name == 'nt':
        for f in generated_files:
            try:
                os.startfile(f)
            except:
                pass


if __name__ == "__main__":
    main()


