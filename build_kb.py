# build_kb.py
# ------------------------------------------------------------
# OBIETTIVO
#   Costruire una Knowledge Base (KB) in OWL a partire dal dataset UCI Heart (Cleveland),
#   eseguire un reasoner OWL per inferire appartenenze a classi (feature "semantiche"),
#   e generare:
#     1) un file OWL con T-Box + A-Box (ontologia + individui popolati)
#     2) un CSV arricchito con nuove colonne binarie inferite dalla KB
#
# MOTIVAZIONE (nel contesto ML+KB)
#   Le feature semantiche rappresentano "background knowledge" (BK) esplicitata tramite regole OWL.
#   In particolare modelliamo condizioni cliniche/indicatori come classi definitorie (EquivalentTo),
#   così che il reasoner possa inferire automaticamente l'appartenenza di ciascun paziente.
#
# INPUT
#   - heart_uci_cleveland_imputed.csv
#     (deve contenere almeno: age, trestbps, chol, fbs, target)
#
# OUTPUT
#   - heart_ontology.owl           : ontologia popolata (T-Box + A-Box)
#   - heart_dataset_enriched.csv   : dataset originale + colonne semantiche (is_*)
#
# NOTE TECNICHE IMPORTANTI
#   - Per Pellet/HermiT serve Java installato.
#   - Le soglie numeriche in OWL (>=) vengono espresse tramite datatype restrictions
#     (ConstrainedDatatype) per ottenere inferenza corretta con Owlready2.
# ------------------------------------------------------------

import time
import pandas as pd
from owlready2 import (
    get_ontology, Thing, DataProperty,
    ConstrainedDatatype,
    sync_reasoner, sync_reasoner_pellet
)

# -----------------------------
# Configurazione I/O
# -----------------------------
INPUT_CSV = "data/heart_uci_cleveland_imputed.csv"     
OUT_ENRICHED_CSV = "data/heart_dataset_enriched.csv"
OUT_OWL = "heart_ontology.owl"

# Elenco delle feature semantiche che vogliamo esportare nel CSV arricchito
# (ogni feature è una colonna binaria 0/1 derivata da inferenza OWL)
SEM_FEATURES = [
    "is_Hypertensive",
    "is_SevereHypertensive",
    "is_Hyperchol",
    "is_HighSugar",
    "is_Older",
    "is_MetabolicSyndromeSuspect",
    "is_SilentHighRisk",
]

# -----------------------------
# Caricamento dataset
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# Controllo minimo delle colonne indispensabili per:
# - popolamento A-Box (valori da assegnare agli individui)
# - regole (classi definite tramite soglie/valori)
required_cols = {"age", "trestbps", "chol", "fbs", "target"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Mancano queste colonne in {INPUT_CSV}: {sorted(missing)}")

# Conversione difensiva dei tipi: in OWL le DataProperty richiedono valori coerenti (int)
df["age"] = df["age"].astype(int)
df["trestbps"] = df["trestbps"].astype(int)
df["chol"] = df["chol"].astype(int)
df["fbs"] = df["fbs"].astype(int)
df["target"] = df["target"].astype(int)

# -----------------------------
# Costruzione ontologia (T-Box)
# -----------------------------
# URI fittizio: sufficiente per un progetto didattico (non richiede hosting reale)
onto = get_ontology("http://example.org/heart_kb.owl")

with onto:
    # Classe principale: paziente (ogni riga del dataset diventa un individuo Patient)
    class Patient(Thing):
        pass

    # Data properties (attributi) per collegare i valori del dataset agli individui
    class hasAge(DataProperty):
        domain = [Patient]
        range = [int]

    class hasBP(DataProperty):  # pressione a riposo (trestbps)
        domain = [Patient]
        range = [int]

    class hasChol(DataProperty):  # colesterolo (chol)
        domain = [Patient]
        range = [int]

    class hasFBS(DataProperty):  # flag glicemia a digiuno (1 se >120 mg/dl, altrimenti 0)
        domain = [Patient]
        range = [int]

    class hasTarget(DataProperty):  # etichetta (0/1) per il task di classificazione
        domain = [Patient]
        range = [int]

    # -----------------------------
    # Classi definite (regole OWL) - "cuore" della KB
    # -----------------------------
    # Le classi seguenti sono definite tramite EquivalentTo:
    # il reasoner può quindi inferire l'appartenenza di un individuo a tali classi
    # in base ai valori assegnati alle DataProperty.

    class Older(Patient):
        # Paziente anziano: age >= 55
        equivalent_to = [
            Patient & hasAge.some(ConstrainedDatatype(int, min_inclusive=55))
        ]

    class Hypertensive(Patient):
        # Ipertensione: trestbps >= 140
        equivalent_to = [
            Patient & hasBP.some(ConstrainedDatatype(int, min_inclusive=140))
        ]

    class SevereHypertensive(Patient):
        # Ipertensione severa: trestbps >= 160
        equivalent_to = [
            Patient & hasBP.some(ConstrainedDatatype(int, min_inclusive=160))
        ]

    class Hyperchol(Patient):
        # Ipercolesterolemia: chol >= 240
        equivalent_to = [
            Patient & hasChol.some(ConstrainedDatatype(int, min_inclusive=240))
        ]

    class HighSugar(Patient):
        # Glicemia alta (flag nel dataset): fbs == 1
        # (value restriction = valore fissato)
        equivalent_to = [
            Patient & hasFBS.value(1)
        ]

    # Classi composte: combinazioni "non banali" per mostrare il valore del ragionamento
    class MetabolicSyndromeSuspect(Patient):
        # Profilo metabolico sospetto: anziano + glicemia alta + colesterolo alto
        equivalent_to = [
            Patient & Older & HighSugar & Hyperchol
        ]

    class SilentHighRisk(Patient):
        # Profilo "silente" ad alto rischio: anziano + ipertensione severa
        equivalent_to = [
            Patient & Older & SevereHypertensive
        ]

# -----------------------------
# Popolamento A-Box (individui)
# -----------------------------
# Per ogni riga del dataset creiamo un individuo Patient e assegnamo le DataProperty.
# Nota Owlready2: le DataProperty sono liste di valori, quindi usiamo [val].
patients = []
for i, row in df.iterrows():
    p = onto.Patient(f"patient_{i}")

    p.hasAge = [int(row["age"])]
    p.hasBP = [int(row["trestbps"])]
    p.hasChol = [int(row["chol"])]
    p.hasFBS = [int(row["fbs"])]
    p.hasTarget = [int(row["target"])]

    patients.append(p)

# -----------------------------
# Ragionamento automatico
# -----------------------------
# Strategia:
#   1) proviamo Pellet (spesso più robusto con datatype restrictions)
#   2) in caso di errore facciamo fallback sul reasoner di default (tipicamente HermiT)
def run_reasoner():
    start = time.time()
    try:
        sync_reasoner_pellet(
            infer_property_values=True,
            infer_data_property_values=True
        )
        used = "pellet"
    except Exception as e:
        print(f"[WARN] Pellet fallito ({type(e).__name__}: {e}). Provo sync_reasoner()...")
        sync_reasoner()
        used = "default"
    elapsed = time.time() - start
    return used, elapsed

reasoner_used, reasoning_time = run_reasoner()
print(f"[INFO] Reasoner usato: {reasoner_used} | tempo: {reasoning_time:.2f}s")

# -----------------------------
# Estrazione feature semantiche
# -----------------------------
# Usiamo INDIRECT_is_a per includere anche le appartenenze inferite (non solo esplicite).
def inferred(instance, cls) -> int:
    return 1 if cls in instance.INDIRECT_is_a else 0

rows = []
for p in patients:
    rows.append({
        "is_Hypertensive": inferred(p, onto.Hypertensive),
        "is_SevereHypertensive": inferred(p, onto.SevereHypertensive),
        "is_Hyperchol": inferred(p, onto.Hyperchol),
        "is_HighSugar": inferred(p, onto.HighSugar),
        "is_Older": inferred(p, onto.Older),
        "is_MetabolicSyndromeSuspect": inferred(p, onto.MetabolicSyndromeSuspect),
        "is_SilentHighRisk": inferred(p, onto.SilentHighRisk),
    })

df_sem = pd.DataFrame(rows)

# Controllo di sanità: se tutte le somme sono 0, probabilmente il reasoning non ha funzionato
print("[INFO] Conteggi feature inferite (devono essere > 0 per almeno alcune):")
print(df_sem.sum().sort_values(ascending=False))

# -----------------------------
# Salvataggio output
# -----------------------------
# Il dataset arricchito concatena il CSV originale con le nuove colonne semantiche.
df_enriched = pd.concat([df.reset_index(drop=True), df_sem], axis=1)
df_enriched.to_csv(OUT_ENRICHED_CSV, index=False)

# Salviamo l'ontologia in RDF/XML (formato standard)
onto.save(file=OUT_OWL, format="rdfxml")

print(f"[OK] Salvato dataset arricchito: {OUT_ENRICHED_CSV} ({len(df_enriched)} righe)")
print(f"[OK] Salvata ontologia: {OUT_OWL}")

