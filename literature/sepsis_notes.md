
# Sepsis: Clinical Notes & Definitions

> **Author:** Lakshya Agarwal
> **Date:** May 2026
> **Source:** Singer et al. (2016), Sepsis-3 Consensus Definitions

---

## 1. Original Definition (Pre-2016): Sepsis = Infection + SIRS

The original definition of sepsis required two things:
1. **Infection** (confirmed or suspected)
2. **SIRS** — Systemic Inflammatory Response Syndrome

### SIRS Criteria
SIRS is defined as the presence of **at least 2** of the following:

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Body Temperature | > 38°C or < 36°C |
| 2 | Heart Rate | > 90 beats per minute |
| 3 | Respiratory Rate | > 20 breaths/min **or** PaCO₂ < 32 mmHg |
| 4 | White Blood Cell Count | > 12,000 cells/mm³ or < 4,000 cells/mm³ or > 10% immature (band) forms |

---

## 2. Why the SIRS-Based Definition Was Problematic

The SIRS criteria were **too broad** — they captured too many patients who did not actually have sepsis:

- Most patients have fevers above 38°C at some point
- Many patients have heart rates above 90 bpm (especially in a clinical setting)
- Leukocyte count > 12,000 cells/mm³ is very common
- Most doctors do not routinely measure respiratory rate

Furthermore, **many non-infectious diseases** can also cause SIRS:
- Trauma
- Burns
- Pancreatitis
- Post-surgical inflammation

This resulted in a **high false positive rate** for sepsis diagnosis using SIRS criteria, making it a poor diagnostic tool in practice.

---

## 3. Sepsis-3 Definition (2016) — Current Standard

In 2016, a new consensus definition was published:

> **"Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection."**
> — Singer et al., JAMA 2016

### Key Differences from Old Definition

| Aspect | Old Definition | Sepsis-3 |
|--------|---------------|----------|
| Focus | Infection + inflammation (SIRS) | Infection + **organ dysfunction** |
| Sensitivity | High (too many false positives) | More specific |
| Clinical emphasis | Inflammatory response | Dysregulated host response |
| Diagnostic tool | SIRS criteria | SOFA score |

This new definition:
- Is more **specific** (focuses on organ dysfunction, not just inflammation)
- Emphasizes the **host response** (which can be dysregulated in sepsis)
- Has led to better understanding, diagnosis, and treatment of sepsis

---

## 4. Clinical Tools for Sepsis Assessment

### 4.1 Full SOFA Score (Sequential Organ Failure Assessment)

The **official tool** for Sepsis-3 diagnosis. Measures dysfunction across **6 organ systems**:

| Component | What It Measures | Key Variables |
|-----------|-----------------|---------------|
| **Respiratory** | Lung function | PaO₂/FiO₂ ratio |
| **Coagulation** | Clotting ability | Platelet count |
| **Liver** | Liver function | Bilirubin |
| **Cardiovascular** | Heart/vessel function | MAP + vasopressor use |
| **Renal** | Kidney function | Creatinine + urine output |
| **Neurological** | Brain function | Glasgow Coma Scale (GCS) |

> **Sepsis-3 diagnosis** = Suspected infection + **SOFA score increase ≥ 2 points**

Each component is scored 0–4 (0 = normal, 4 = severe dysfunction). Total SOFA ranges from 0–24.

### 4.2 qSOFA Score (Quick SOFA)

A simplified bedside tool with only **3 criteria**:

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | GCS (Neurological) | < 15 |
| 2 | Systolic Blood Pressure | < 100 mmHg |
| 3 | Respiratory Rate | > 22 breaths/minute |

**Score ≥ 2 = high risk of poor outcome**

> ⚠️ **IMPORTANT: qSOFA is NOT a diagnostic tool for sepsis.**
> It is only a **risk assessment/screening tool** to identify patients who should be evaluated further.

---

## 5. Clinical Reality Check

### Why qSOFA Has Limitations
- **Not sensitive enough** to be used as a diagnostic or screening tool
- Can **miss many sepsis patients** (especially early-stage)
- Many institutions still **refuse to use qSOFA** and stick to the older SIRS-based criteria
	- They are aware SIRS is imperfect, but find it more practical and sensitive

### The Bottom Line
> **No scoring system in the world can replace solid clinical thinking.**

Scoring systems (SOFA, qSOFA, SIRS, NEWS) are tools — not replacements for clinical judgment. They assist decision-making but should always be interpreted in clinical context.

---

## 6. Relevance to This Project

| Concept | How It Applies to Our Work |
|---------|---------------------------|
| **SIRS definition** | Historical context; we do NOT use this for labeling |
| **Sepsis-3 definition** | Our outcome definition — what we predict |
| **Full SOFA score** | How we **label** sepsis cases in MIMIC-IV (SOFA ≥ 2 + infection) |
| **qSOFA** | Our **clinical baseline to beat** — our model should outperform qSOFA |
| **SOFA components** | What lab values we extract from MIMIC-IV to compute SOFA |

### Label Construction (How We Identify Sepsis in MIMIC-IV)

A patient is labeled as **sepsis-3 positive** if they meet both:
1. **Suspected infection**: Concurrent antibiotic administration + microbiological culture order
2. **Organ dysfunction**: SOFA score increase ≥ 2 points from baseline

The **prediction task**: Given 12 hours of vitals and labs, predict if a patient will meet Sepsis-3 criteria within the **next 6 hours**.

---

## 7. SOFA Score Computation from MIMIC-IV

When we build the cohort extraction pipeline, we will compute SOFA from these MIMIC-IV variables:

| SOFA Component | MIMIC-IV Source | Variable |
|----------------|----------------|----------|
| Respiratory | chartevents / labevents | PaO₂, FiO₂ |
| Coagulation | labevents | Platelet count |
| Liver | labevents | Bilirubin |
| Cardiovascular | chartevents + inputevents | MAP, vasopressor flag |
| Renal | labevents + outputevents | Creatinine, urine output |
| Neurological | chartevents | Glasgow Coma Scale |

---

## 8. Clinical Guidance & Practical Notes

- qSOFA is a quick risk flag — use it to prompt further evaluation, not to diagnose.
- SIRS criteria remain useful for sensitivity in screening workflows, but are nonspecific.
- Always combine scoring outputs with clinical context, cultures, and imaging when deciding management.

---

## 9. Key References

- **Singer M et al.** (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810. https://doi.org/10.1001/jama.2016.0287
- **Seymour CW et al.** (2016). Assessment of Clinical Criteria for Sepsis. *JAMA*, 315(8), 762–774.
- **Shankar-Hari M et al.** (2016). Developing a New Definition and Assessing New Clinical Criteria for Septic Shock. *JAMA*, 315(8), 775–787.

---

*Last updated: May 2026 | Lakshya Agarwal | JK Lakshmipat University*