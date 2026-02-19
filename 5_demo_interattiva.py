import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# DEMO INTERATTIVA (TUTTO IN ITALIANO)
#
# SCOPO DIDATTICO
#   Fornire una dimostrazione "utente finale" del sistema neuro-simbolico:
#   - l'utente inserisce pochi dati (stile screening)
#   - il sistema applica regole coerenti con la KB (feature semantiche)
#   - un modello ML (RandomForest) produce una probabilità di rischio
#   - viene mostrata una spiegazione leggibile (quali pattern KB si attivano)
#
# IMPORTANTE
#   Questa demo serve a mostrare: integrazione ML + regole/feature semantiche (BK).
#
# DIPENDENZE
#   Richiede "heart_dataset_enriched.csv" generato da build_kb.py
# ------------------------------------------------------------


# -----------------------------
# Soglie / regole della KB (coerenti con build_kb.py)
# -----------------------------
SOGLIA_ETA = 55
SOGLIA_PRESSIONE = 140
SOGLIA_PRESSIONE_SEVERA = 160
SOGLIA_COLESTEROLO = 240


def inferenza_feature_semantiche(eta: int, pressione: int, colesterolo: int, fbs: int) -> dict:
    """
    Gemello digitale delle regole OWL definite nella KB (build_kb.py).

    Razionale:
      In fase di demo vogliamo un feedback immediato e spiegabile.
      Qui replichiamo le stesse condizioni logiche della KB,
      generando feature binarie con gli stessi nomi del CSV arricchito.
    """
    is_older = int(eta >= SOGLIA_ETA)
    is_hypertensive = int(pressione >= SOGLIA_PRESSIONE)
    is_severe_hypertensive = int(pressione >= SOGLIA_PRESSIONE_SEVERA)
    is_hyperchol = int(colesterolo >= SOGLIA_COLESTEROLO)
    is_highsugar = int(fbs == 1)

    # Classi composte 
    is_metabolic = int(is_older and is_highsugar and is_hyperchol)
    is_silent_highrisk = int(is_older and is_severe_hypertensive)

    return {
        "is_Hypertensive": is_hypertensive,
        "is_SevereHypertensive": is_severe_hypertensive,
        "is_Hyperchol": is_hyperchol,
        "is_HighSugar": is_highsugar,
        "is_Older": is_older,
        "is_MetabolicSyndromeSuspect": is_metabolic,
        "is_SilentHighRisk": is_silent_highrisk,
    }


class SistemaCDSS:
    def __init__(self):
        """
        Inizializza la demo:
          1) carica il dataset arricchito
          2) verifica presenza colonne necessarie
          3) addestra un RandomForest sul dataset (base + semantiche)
        """
        print("\n🏥 INIZIALIZZAZIONE SISTEMA CDSS (Demo interattiva)...")

        # Controllo preliminare: la demo dipende dal dataset arricchito
        if not os.path.exists("data/heart_dataset_enriched.csv"):
            print("❌ Errore: manca 'heart_dataset_enriched.csv'. Esegui prima build_kb.py.")
            raise SystemExit(1)

        df = pd.read_csv("data/heart_dataset_enriched.csv")

        # Target richiesto
        if "target" not in df.columns:
            print("❌ Errore: nel CSV non trovo la colonna 'target'.")
            print(f"   Colonne disponibili: {list(df.columns)}")
            raise SystemExit(1)

        # Conversione difensiva del target
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

        # Feature low-cost (screening) usate nella demo
        self.feature_base = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang"]

        # Feature semantiche con naming coerente con build_kb.py
        self.feature_semantiche = [
            "is_Hypertensive",
            "is_SevereHypertensive",
            "is_Hyperchol",
            "is_HighSugar",
            "is_Older",
            "is_MetabolicSyndromeSuspect",
            "is_SilentHighRisk",
        ]

        # Robustezza: se qualche colonna semantica manca, la ricostruiamo localmente
        mancanti = [c for c in self.feature_semantiche if c not in df.columns]
        if mancanti:
            print(f"⚠️  Nel CSV mancano alcune feature semantiche {mancanti}. Le ricostruisco con le regole della KB...")
            sem_rows = []
            for _, r in df.iterrows():
                sem_rows.append(
                    inferenza_feature_semantiche(
                        int(r["age"]), int(r["trestbps"]), int(r["chol"]), int(r["fbs"])
                    )
                )
            df_sem = pd.DataFrame(sem_rows)
            for c in self.feature_semantiche:
                if c not in df.columns:
                    df[c] = df_sem[c].astype(int)

        # Verifica finale di tutte le colonne richieste dal modello
        needed = self.feature_base + self.feature_semantiche + ["target"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print("❌ ERRORE: mancano colonne necessarie per la demo.")
            print("   Mancano:", missing)
            print("   Colonne disponibili:", list(df.columns))
            raise SystemExit(1)

        # Addestramento modello:
        # scopo demo = ottenere una probabilità di rischio, non massimizzare tuning
        print("🧠 Addestramento del modello ML sui dati storici...", end=" ")
        X = df[self.feature_base + self.feature_semantiche]
        y = df["target"]

        self.modello = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.modello.fit(X, y)
        print("Fatto. ✅")


    # -----------------------------
    # Input robusto: validazione e menù
    # -----------------------------
    def chiedi_intero(self, prompt, min_val, max_val):
        """Chiede un intero tra min_val e max_val (inclusi)."""
        while True:
            try:
                val = int(input(f"{prompt} ({min_val}-{max_val}): ").strip())
                if min_val <= val <= max_val:
                    return val
                print(f"⚠️  Inserisci un valore tra {min_val} e {max_val}.")
            except ValueError:
                print("⚠️  Inserisci un numero intero valido.")

    def chiedi_scelta(self, prompt, opzioni):
        """Mostra opzioni numerate e ritorna la scelta valida."""
        print(prompt)
        for key, descr in opzioni.items():
            print(f"   [{key}] {descr}")
        while True:
            try:
                val = int(input("   > Scelta: ").strip())
                if val in opzioni:
                    return val
                print("⚠️  Opzione non valida.")
            except ValueError:
                print("⚠️  Inserisci il numero dell'opzione.")

    # -----------------------------
    # Valutazione singolo paziente
    # -----------------------------
    def valuta_paziente(self):
        """
        Flusso della demo:
          1) acquisizione input
          2) inferenza feature semantiche (replica KB)
          3) predizione probabilistica ML
          4) stampa spiegazione e messaggio finale (italiano)
        """
        print("\n" + "=" * 66)
        print("🩺  NUOVA VALUTAZIONE PAZIENTE (SCREENING)")
        print("=" * 66)

        eta = self.chiedi_intero("1) Età", 18, 100)
        sesso = self.chiedi_scelta("2) Sesso biologico:", {1: "Maschio", 0: "Femmina"})

        cp = self.chiedi_scelta(
            "3) Tipo di dolore toracico (cp):",
            {
                1: "Angina tipica (dolore compatibile con ischemia cardiaca)",
                2: "Angina atipica (dolore non completamente tipico)",
                3: "Dolore non-anginoso (più probabilmente non cardiaco)",
                4: "Asintomatico (nessun dolore toracico riferito)",
            },
        )

        pressione = self.chiedi_intero("4) Pressione a riposo (trestbps, mmHg)", 80, 220)
        colesterolo = self.chiedi_intero("5) Colesterolo (chol, mg/dl)", 100, 600)

        fbs = self.chiedi_scelta(
            "6) Glicemia a digiuno > 120 mg/dl? (fbs)",
            {1: "Sì (alta)", 0: "No (normale)"},
        )

        restecg = self.chiedi_scelta(
            "7) Esito ECG a riposo (restecg):",
            {
                0: "Normale",
                1: "Anomalia ST-T (possibili alterazioni della ripolarizzazione)",
                2: "Ipertrofia ventricolare sinistra (criteri ECG)",
            },
        )

        thalach = self.chiedi_intero("8) Frequenza cardiaca massima raggiunta (thalach)", 60, 220)
        exang = self.chiedi_scelta("9) Angina indotta da sforzo? (exang)", {1: "Sì", 0: "No"})

        # Inferenza semantica: genera le stesse colonne is_* usate nel training
        sem = inferenza_feature_semantiche(eta, pressione, colesterolo, fbs)

        # Vettore input per il modello (ordine coerente con X in training)
        base_vec = [eta, sesso, cp, pressione, colesterolo, fbs, restecg, thalach, exang]
        sem_vec = [sem[c] for c in self.feature_semantiche]
        x = np.array(base_vec + sem_vec, dtype=float).reshape(1, -1)

        # Probabilità della classe positiva (target=1)
        prob = float(self.modello.predict_proba(x)[0][1])

        # Report leggibile
        print("\n" + "-" * 66)
        print("📄 REPORT (Neuro-Simbolico / Spiegabile)")
        print("-" * 66)
        print("🔎 Risultati della componente di ragionamento (feature semantiche inferite):")

        def mostra(flag, testo, icona_true="✅", icona_false="❌"):
            """Stampa un indicatore booleano in modo leggibile."""
            print(f"   {icona_true if flag else icona_false} {testo}")

        mostra(sem["is_Older"], f"Older: età ≥ {SOGLIA_ETA}")
        mostra(sem["is_Hypertensive"], f"Hypertensive: pressione ≥ {SOGLIA_PRESSIONE}")
        mostra(sem["is_SevereHypertensive"], f"SevereHypertensive: pressione ≥ {SOGLIA_PRESSIONE_SEVERA}")
        mostra(sem["is_Hyperchol"], f"Hyperchol: colesterolo ≥ {SOGLIA_COLESTEROLO}")
        mostra(sem["is_HighSugar"], "HighSugar: glicemia a digiuno alta (fbs=1)")
        mostra(sem["is_MetabolicSyndromeSuspect"], "MetabolicSyndromeSuspect: Older & HighSugar & Hyperchol", "🚨", "—")
        mostra(sem["is_SilentHighRisk"], "SilentHighRisk: Older & SevereHypertensive", "🚨", "—")

        print("\n📊 Output del modello ML:")
        print(f"   Probabilità di rischio (classe positiva): {prob:.1%}")

        # Interpretazione dimostrativa (soglie scelte per demo, non cliniche)
        print("\n🧾 Interpretazione (dimostrativa, non diagnostica):")
        if prob >= 0.75:
            print("   🔴 RISCHIO MOLTO ALTO → consigliato approfondimento immediato (visita/esami).")
        elif prob >= 0.50:
            print("   🟠 RISCHIO MEDIO-ALTO → consigliata valutazione clinica e ulteriori accertamenti.")
        else:
            print("   🟢 RISCHIO BASSO → monitoraggio e prevenzione (stile di vita / controlli periodici).")

        # Caso esplicativo: paziente asintomatico ma rischio non trascurabile
        if cp == 4 and prob >= 0.50:
            print("\n   📌 Nota: paziente ASINTOMATICO (cp=4) ma rischio non trascurabile.")
            if sem["is_SilentHighRisk"]:
                print("      La KB evidenzia un pattern di 'rischio silente' (SilentHighRisk).")

        print("-" * 66)


if __name__ == "__main__":
    sistema = SistemaCDSS()
    while True:
        sistema.valuta_paziente()
        cont = input("\nVuoi valutare un altro paziente? (s/n): ").strip().lower()
        if cont not in ("s", "si", "sì", "y", "yes"):
            print("\n✅ Uscita dal sistema. Buon lavoro!\n")
            break
