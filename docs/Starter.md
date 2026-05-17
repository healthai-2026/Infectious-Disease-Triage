## AI-Enabled Clinical Triage and Remote Monitoring for Infectious Disease Care

### Overview

Early recognition of clinical deterioration is one of the most impactful interventions in critical care. For infectious diseases, pneumonia, sepsis, influenza, and other acute infections, identifying patients at risk of deterioration even a few hours earlier can meaningfully reduce mortality. Yet most hospitals still rely on threshold-based scoring systems (qSOFA, NEWS, MEWS) that are poorly sensitive for early deterioration and contribute significantly to alert fatigue.

AI-driven early warning systems that continuously analyze electronic health records (EHRs), physiological time-series, and clinical notes offer a compelling alternative. Recent work has shown that deep learning models operating on ICU vital signs and laboratory data can predict sepsis onset 6–12 hours before clinical recognition, with performance competitive against experienced clinicians. Integrating unstructured clinical notes via language models adds further predictive signal. Remote patient monitoring powered by wearable sensors extends this capability outside the hospital.

Your task is to build and evaluate an AI-based early warning or triage system for a clinically relevant deterioration event, sepsis onset, in-hospital mortality, or ICU transfer, using publicly available critical care EHR data. The work should emphasize prediction performance, explainability of outputs, and awareness of real-world deployment constraints such as missing data, class imbalance, and alert calibration.

---

### Datasets

The following are the standard benchmarks in this area. Access to most requires a free PhysioNet credentialing process, which you should initiate early.

- **MIMIC-IV** (PhysioNet): De-identified EHR data from ICU admissions at a major US hospital. Includes structured vitals, labs, medications, and procedures. The primary dataset for this project.
- **MIMIC-IV-Note** (PhysioNet): De-identified clinical notes (discharge summaries, nursing notes, radiology reports) matched to MIMIC-IV patients.
- **eICU Collaborative Research Database** (PhysioNet): Multi-center ICU data from 208 hospitals. Useful for evaluating generalizability across sites.
- **PhysioNet/CinC Challenge 2019**: Curated hourly ICU time-series (vitals + labs) with Sepsis-3 labels. An excellent benchmark for sepsis onset prediction.
- **MIMIC-IV-ECG** (PhysioNet): 12-lead ECG waveforms linked to MIMIC-IV clinical records.
- **MIMIC Waveform Database**: Bedside monitor waveforms (ECG, PPG, arterial blood pressure) matched to MIMIC-III patients.

---

### Papers to Read

Start with these and follow citations to build deeper understanding. This is not an exhaustive list.

**Foundational:**
- Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," *Scientific Data* (2023).
- Harutyunyan et al., "Multitask learning and benchmarking with clinical time series data," *Scientific Data* (2019). [MIMIC-III benchmark tasks, in-hospital mortality, decompensation, length of stay]
- Futoma et al., "An improved multi-output Gaussian process RNN with real-time validation for early sepsis detection," *MLHC* (2017).
- Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv (2017).

**Recent:**
- Wang et al., "Development and prospective implementation of a large language model based system for early sepsis prediction," *npj Digital Medicine* (2025). [COMPOSER-LLM]
- Chang et al., "A federated learning framework with knowledge graph and temporal transformer for early sepsis prediction in multi-center ICUs," arXiv:2603.15651 (2025).
- Al-Juhani et al., "Advances in data-driven early warning systems for sepsis recognition in emergency care," *Cureus* (2025). [Systematic review, read this early]
- Thirunavukarasu et al., "Large language models in medicine," *Nature Medicine* (2023).
- Guevara et al., "Large language models to identify social determinants of health in EHRs," *npj Digital Medicine* (2024).
- Mahajan et al., "Wearable AI to enhance patient safety and clinical decision-making," *npj Digital Medicine* (2025).

---

### Suggested Learning Topics

You are responsible for acquiring the necessary background. The following areas are directly relevant to this project.

- ICU clinical concepts: sepsis definitions (Sepsis-3, SOFA score), vital signs, standard lab panels
- Time-series preprocessing: irregular sampling, forward-fill imputation, normalization, windowing
- Sequence models for clinical data: LSTMs, GRUs, Temporal Convolutional Networks, Transformers (PatchTST, iTransformer)
- Clinical NLP: BERT-based models fine-tuned on clinical text (BioClinicalBERT, ClinicalBERT)
- Multimodal fusion of structured and unstructured clinical data
- Federated learning concepts: FedAvg, privacy-preserving distributed training
- Evaluation under class imbalance: AUROC, AUPRC, calibration, alert fatigue metrics
- SHAP and attention-based explainability for time-series data
- Deployment considerations: latency, threshold selection, false alarm rate trade-offs

---

### Expected Outcomes (12-Week Plan)

The 12 weeks are structured around four phases. Specific milestones should be defined in discussion with your mentor.

**Weeks 1–3: Foundation**
Complete the PhysioNet credentialing and gain access to MIMIC-IV. Conduct a focused literature review. Define your clinical prediction task (e.g., sepsis-3 onset at 6-hour horizon) and build the cohort extraction pipeline with proper label construction. Deliver a written cohort definition document.

**Weeks 4–6: Baseline Models**
Implement at least two baselines: a classical ML model (logistic regression or XGBoost on handcrafted features) and a temporal deep learning model (LSTM or Transformer). Evaluate against published NEWS/qSOFA thresholds. Document performance and failure modes.

**Weeks 7–9: Extension**
Add a second input modality (clinical notes via pre-trained embeddings, or waveform features) and measure its contribution via ablation. Optionally, simulate a federated learning scenario using MIMIC-IV and eICU splits. Implement one explainability method and analyze top predictive features.

**Weeks 10–12: Research Output**
Consolidate findings, characterize model behavior under missing data and class imbalance, and write a research report in paper format. Final deliverables: clean, documented code repository and a written report (journal paper style).
