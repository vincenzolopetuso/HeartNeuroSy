# HeartNeuroSy: Un Sistema Ibrido Neuro-Simbolico 

**Autore:** Vincenzo Lopetuso - 797844
**Corso:** Ingegneria della Conoscenza (ICon) - A.A. 2025/2026

## Descrizione del Progetto
HeartNeuroSy è un Clinical Decision Support System (CDSS) ibrido che integra **Ontologie (Knowledge Base in OWL)** e modelli di **Machine Learning (Apprendimento Supervisionato)**. 
L'obiettivo è prevedere il rischio di malattie cardiache a partire dal dataset *UCI Heart Disease (Cleveland)*, dimostrando come il ragionamento simbolico (es. l'inferenza di pazienti con Sindrome Metabolica o Rischio Silente) possa migliorare le performance predittive, specialmente in scenari di screening clinico con scarsi dati a disposizione (low-data).

## Struttura della Repository
- `data/`: Contiene i dataset originali e quelli arricchiti dalla KB.
- `docs/`: Contiene la relazione dettagliata del progetto (`Relazione_HeartNeuroSy.pdf`).
- `images/`: Contiene i grafici (PNG) di valutazione delle performance.
- `build_kb.py`: Script per la creazione dell'ontologia e inferenza tramite reasoner (Pellet/HermiT).
- `2_model_comparison.py`: Confronto tra modelli su dataset base vs arricchito.
- `3_screening_test.py`: Simulazione di uno scenario clinico a basso costo (screening).
- `4_plot_results.py`: Generazione dei grafici delle metriche di valutazione.
- `5_demo_interattiva.py`: Demo interattiva da terminale.

## Installazione ed Esecuzione
1. Clonare il repository.
2. Installare le dipendenze:
   ```bash
   pip install -r requirements.txt

