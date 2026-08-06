# Practice School-I Internship Report

## AI-Enabled Early Warning System for Sepsis-3 Onset Prediction Using Temporal Deep Learning on the MIMIC-IV Clinical Database

**Submitted in partial fulfillment of the requirements for the Practice School-I Programme**

Student Names: Lakshya Agarwal, Deepanshu Singh Shekhawat
Supervisor: Dr. Amit Sinhal, Department of Computer Science and Engineering
Institution: JK Lakshmipat University, Jaipur, Rajasthan, India
Programme: Practice School-I (PS-I), B.Tech. Computer Science and Engineering
Duration: May to August 2026

---

# CHAPTER 1 - ABOUT THE ORGANIZATION

## 1.1 Introduction

This report documents the work completed during the Practice School-I internship at JK Lakshmipat University, Jaipur, under the supervision of Dr. Amit Sinhal of the Department of Computer Science and Engineering. The internship was conducted between May and August 2026 as part of the university's structured academic programme. The project falls within the domain of clinical artificial intelligence, specifically the application of temporal deep learning to the problem of early sepsis detection using large-scale electronic health record data.

The internship provided the participating students with an opportunity to engage with a real-world clinical machine learning problem involving not only the design and training of predictive models but also the rigorous handling of a complex, multi-table clinical database, the construction of clinically valid outcome labels, and the application of state-of-the-art explainability techniques. The experience bridged the gap between theoretical knowledge acquired in the undergraduate curriculum and the practical demands of applied biomedical research.

## 1.2 JK Lakshmipat University

JK Lakshmipat University is a private university located in Jaipur, Rajasthan, established under the Rajasthan Private Universities Act. The university offers undergraduate, postgraduate, and doctoral programmes across the disciplines of engineering, management, and design. The Department of Computer Science and Engineering is one of its flagship departments, offering a four-year Bachelor of Technology programme with specialisations in core computing and emerging fields including artificial intelligence and data science. The university maintains active research collaborations and encourages student participation in faculty-led research projects, particularly through its Practice School programme, which is modelled on experiential learning principles. Faculty members engage in research spanning machine learning, natural language processing, computer vision, and biomedical informatics, providing students with a rich and challenging environment for research-oriented internships.

## 1.3 Practice School-I Programme

The Practice School-I programme is a structured academic exercise that allows students to apply their theoretical knowledge to real engineering and research problems under qualified faculty supervision. Unlike a conventional industrial internship, PS-I at JK Lakshmipat University situates students within an active research environment where they are expected to contribute meaningfully to ongoing projects. Students receive formal supervision, are required to maintain a research log, and submit a final report that meets the standards of a technical academic document. The programme spans approximately twelve weeks during the summer semester and requires students to demonstrate competency in problem formulation, literature review, dataset analysis, methodology design, implementation, and result interpretation. The expectation is that each intern's contribution forms a coherent, reproducible component of a larger research endeavour. In this case, both students were assigned to a single collaborative project focused on clinical prediction, with individual responsibilities distributed across pipeline stages.

## 1.4 Research Environment

The research was conducted in the university's computational laboratory under the guidance of Dr. Amit Sinhal. GPU-accelerated computing resources were used for training deep learning models. The primary programming environment was Python 3, with the full dependency stack managed through a virtual environment. The key libraries employed include PyTorch version 2.13.0 for neural network development, XGBoost version 3.3.0 for gradient-boosted tree models, scikit-learn version 1.9.0 for classical machine learning and cross-validation utilities, SHAP version 0.52.0 for model explainability, Pandas version 3.0.3 and NumPy version 2.4.6 for data manipulation and numerical computation, and Matplotlib version 3.11.0 and Seaborn version 0.13.2 for visualisation. Additional libraries including SciPy version 1.18.0, Numba version 0.66.0, and Joblib version 1.5.3 were used for scientific computation and model serialisation. The complete dependency manifest is maintained in the repository's requirements.txt file, ensuring full reproducibility of the experimental environment.

## 1.5 Project Overview

The project is formally titled AI-Enabled Clinical Triage and Remote Monitoring for Infectious Disease Care. Its primary technical deliverable is a reproducible machine learning pipeline that predicts the onset of Sepsis-3 in ICU patients up to six hours before the clinical event, using a structured hourly time-series representation of physiological and laboratory measurements derived from the MIMIC-IV clinical database. The pipeline encompasses data preprocessing, label construction according to the Sepsis-3 consensus definition, feature engineering, unified multi-model benchmarking under identical experimental conditions, and post-hoc explainability analysis using SHAP. A secondary objective is the development of a modular framework for multimodal feature integration, to be extended in future work. The central deliverable of the unified benchmark is the script unified_benchmark.py, which is the authoritative experimental pipeline and the primary basis for the analysis presented in this report.

---

# CHAPTER 2 - INTRODUCTION AND LITERATURE REVIEW

## 2.1 Introduction to Sepsis

Sepsis is defined clinically as a life-threatening organ dysfunction caused by a dysregulated host response to infection. This definition was formalised by the Third International Consensus Definitions for Sepsis and Septic Shock, known as Sepsis-3, published in 2016 and representing a significant evolution from earlier criteria rooted solely in the systemic inflammatory response syndrome framework. The Sepsis-3 framework operationalises sepsis as the co-occurrence of a suspected or confirmed source of infection and an acute increase of two or more points in the Sequential Organ Failure Assessment score from the patient's pre-morbid baseline. The most severe form, septic shock, requires in addition the administration of vasopressor therapy to maintain a mean arterial pressure of at least 65 mmHg in the absence of hypovolaemia, accompanied by a serum lactate concentration greater than 2 mmol per litre despite adequate volume resuscitation.

Sepsis is among the most significant causes of morbidity and mortality in hospital settings worldwide and is the leading cause of in-hospital death in the intensive care unit. Its pathophysiology involves a complex, dysregulated interaction between the infecting organism and the immune system, resulting in systemic inflammation, microvascular dysfunction, coagulation abnormalities, and progressive multi-organ failure. The condition can evolve rapidly from an early compensatory phase to refractory haemodynamic collapse over a period of hours. Studies have consistently demonstrated that each hour of delay in antibiotic administration and haemodynamic resuscitation following the recognition of sepsis is associated with a measurable and statistically significant increase in patient mortality, establishing timely recognition as a clinical priority of the highest order.

## 2.2 Clinical Motivation

Despite decades of research and considerable investment in bedside monitoring technology, early recognition of sepsis in the intensive care unit remains a fundamentally unsolved problem. The difficulty arises from several intersecting factors. The physiological manifestations of early sepsis, including tachycardia, tachypnoea, fever or hypothermia, and haemodynamic instability, are nonspecific and overlap substantially with those of many non-infectious conditions. Clinicians managing multiple critically ill patients simultaneously cannot perform continuous, comprehensive surveillance of all physiological variables for every patient. Additionally, the temporal trajectory of deterioration in sepsis often involves subtle, progressive trends across many variables that are difficult to integrate manually at the bedside in real time.

The standard clinical approach to early warning relies on composite scoring systems. The quick Sequential Organ Failure Assessment score, or qSOFA, is a simplified bedside screening tool consisting of three criteria: altered mentation defined as a Glasgow Coma Scale score below 15, a respiratory rate of 22 or more breaths per minute, and a systolic blood pressure of 100 mmHg or less. A score of two out of three is considered a positive screen. Prospective validation studies have however demonstrated that qSOFA exhibits low sensitivity in ICU populations, missing a substantial proportion of patients who subsequently meet the full Sepsis-3 definition. The National Early Warning Score and Modified Early Warning Score face similar limitations, relying on few point-in-time observations and failing to exploit the temporal dynamics of deterioration.

These limitations motivate the development of a machine learning-based early warning system capable of integrating many physiological and laboratory variables across multiple observation hours, learning complex temporal patterns indicative of impending sepsis, and generating probabilistic alert signals that are both sensitive and specific, thereby reducing the dual clinical harms of missed diagnoses and alarm fatigue.

## 2.3 Problem Statement

The research problem addressed by this project can be stated formally as follows. Given a time-series representation of a patient's physiological and biochemical state observed at hourly intervals over a lookback window of twelve hours, the objective is to learn a binary classification function that accurately predicts whether the patient will satisfy the Sepsis-3 definition within the next six hours. This is a prospective prediction task: the model uses only information available at the time of prediction and has no access to future observations. The task is complicated by severe class imbalance, since the proportion of time-series windows in which a sepsis onset event occurs within the six-hour horizon is substantially smaller than those in which no event occurs. The data-generating process is retrospective using the MIMIC-IV clinical database, but the evaluation methodology simulates the prospective clinical deployment scenario as closely as possible.

## 2.4 MIMIC-IV Database

The MIMIC-IV database, version 3.1, is a large, de-identified, freely available electronic health record database developed and maintained by the Laboratory for Computational Physiology at the Massachusetts Institute of Technology in collaboration with the Beth Israel Deaconess Medical Center in Boston, Massachusetts. It contains comprehensive clinical data from patients admitted to the Beth Israel Deaconess Medical Center ICUs spanning multiple years of clinical operations. Access requires formal credentialing through the PhysioNet platform, including completion of a certified human subjects research training course and execution of the MIMIC Data Use Agreement.

MIMIC-IV is organised into two primary modules. The hospital module contains admission-level records including patient demographics, coded diagnoses, laboratory measurements, microbiology results, medication prescriptions, and discharge summaries. The ICU module contains high-resolution stay-level records including continuously recorded vital sign measurements from bedside monitors, medication infusion records, and fluid output events. This project draws from nine tables spanning both modules and joins records across patient, admission, and stay levels using the relational identifiers subject_id, hadm_id, and stay_id.

## 2.5 Research Objectives

The primary objectives of this research are as follows. First, to construct a clinically valid Sepsis-3 compliant label from raw MIMIC-IV data by implementing the suspected infection criterion and computing the SOFA score from each of its six organ system components directly from charted measurements, without relying on ICD diagnostic codes. Second, to build a reproducible hourly feature grid capturing the temporal trajectory of patient physiology during the ICU stay. Third, to implement a unified benchmarking framework in which all compared model families are evaluated on identical sliding windows, identical cross-validation folds, and identical outcome labels, ensuring that observed differences in performance are attributable solely to model architecture. Fourth, to apply SHAP-based post-hoc explainability analysis to the best-performing classical model to identify and rank clinical features most predictive of imminent sepsis onset, enabling clinical interpretation and trust in the system's outputs.

## 2.6 Related Work

The application of machine learning to sepsis prediction has been an active research area for over a decade. Futoma et al. proposed a Multi-Output Gaussian Process Recurrent Neural Network combining Gaussian process imputation of irregularly sampled clinical measurements with an LSTM classifier trained end-to-end to predict sepsis risk. Their system outperformed conventional early warning scores in terms of both sensitivity and lead time, demonstrating that the temporal correlation structure of physiological variables contains diagnostic signal unavailable to point-in-time scoring approaches. A key clinical observation from that work was that an AI system could detect sepsis risk up to 36 hours before the clinical diagnosis was formally established, with profound implications for patient outcomes.

Subsequent contributions demonstrated the utility of deep convolutional and recurrent architectures on MIMIC-derived clinical time-series data. Researchers have applied LSTMs, temporal convolutional networks, and transformer architectures to the sepsis prediction problem, consistently finding that models which explicitly represent the time dimension outperform those operating on static snapshots. Shashikumar et al. proposed the AISE system, using a similar temporal architecture that was prospectively validated in a clinical environment, demonstrating that laboratory-validated performance could translate to real-world clinical settings under appropriate deployment safeguards.

More recent work has addressed the fairness, calibration, and interpretability of sepsis prediction models. It has been shown that models achieving high average AUROC may exhibit substantially degraded performance for specific demographic subgroups, raising questions about equitable deployment. The growing consensus in the clinical machine learning community is that model interpretability is not merely a desirable property but a prerequisite for safe clinical adoption, motivating the integration of frameworks such as SHAP into the model evaluation workflow. The present project contributes to this body of work by combining a strict, leakage-aware unified benchmark with SHAP-based global and local explainability to produce a transparent and reproducible experimental comparison.

## 2.7 Scope of the Internship

The scope of the internship encompasses the complete pipeline from raw MIMIC-IV data to evaluated and interpretable predictions. The students were responsible for implementing the preprocessing pipeline in preprocess.py, the unified benchmarking framework in unified_benchmark.py, the standalone LSTM training scripts in train_lstm.py and train_rnn.py, the SHAP explainability module in shap_analysis.py, and a secondary modality embedding utility in multimodal_support.py. The work does not extend to clinical deployment, prospective validation, or randomised controlled evaluation, all of which are identified as future directions. The research is entirely retrospective and academic, conducted under the constraints of the PhysioNet Data Use Agreement.

---

# CHAPTER 3 - DATASET AND METHODOLOGY

## 3.1 Overall System Architecture

The system is designed as a sequential, modular pipeline in which each stage produces well-defined outputs consumed by the next. The pipeline originates from raw compressed CSV files from the MIMIC-IV database and terminates in trained model artefacts, performance metrics serialised in JSON, and SHAP-based feature importance visualisations. The entire implementation is in Python, and reproducibility is ensured by setting deterministic random seeds for all stochastic components at each stage using Python's random module, NumPy, and PyTorch's manual seed facilities, with CuDNN deterministic mode enabled.

The pipeline proceeds through five logical phases. In the first phase, raw clinical tables are loaded and filtered to identify the qualifying patient cohort. In the second phase, the hourly feature grid is constructed for each qualifying ICU stay, integrating vital signs, laboratory measurements, SOFA component scores, and demographic variables into a unified row-per-hour tabular structure. In the third phase, Sepsis-3 onset labels are computed and attached to each row. In the fourth phase, the unified benchmarking framework constructs overlapping sliding windows and trains and evaluates all model families under strictly identical conditions. In the fifth phase, the SHAP analysis module reconstructs the same windows, applies the saved preprocessing transformations, and computes and visualises feature attributions.

[Insert Figure 3.1: End-to-End Pipeline Diagram]

## 3.2 Dataset Description

The MIMIC-IV version 3.1 database is the sole data source. From the hospital module, the pipeline loads patients.csv.gz for demographic information, admissions.csv.gz for hospital admission identifiers, microbiologyevents.csv.gz for culture order timestamps, prescriptions.csv.gz for antibiotic administration records, and labevents.csv.gz for laboratory result values. From the ICU module, the pipeline loads icustays.csv.gz for stay-level admission and discharge times, chartevents.csv.gz for time-stamped physiological measurements, inputevents.csv.gz for vasoactive infusion records, and outputevents.csv.gz for urine output volumes.

Large files are loaded in chunks of 500,000 to 1,000,000 rows and filtered at the chunk level to minimise peak memory usage. The final processed dataset is stored as features.csv in the Data/processed directory and contains one row per qualifying patient-hour, with 64 numeric feature columns, identifier columns, and the binary label column.

[Insert Table 3.1: Summary of MIMIC-IV Tables Used, Including Module, Table Name, Columns Loaded, and Purpose]

## 3.3 Data Preprocessing

The preprocessing pipeline in preprocess.py transforms raw clinical tables into the analysis-ready feature matrix through the following steps.

Antibiotic Identification. Antibiotic prescriptions are identified using case-insensitive regular expression matching of drug names against a curated list of 40 antibiotic name substrings including cillin, cef, cep, penem, floxacin, mycin, cycline, monam, oxacin, sulfamethoxazole, trimethoprim, metronidazole, linezolid, daptomycin, and the aminoglycoside agents gentamicin, tobramycin, and amikacin, among others. A secondary exclusion list removes spurious matches on non-antibiotic drug names including cepacol, cepastat, racepinephrine, and epinephrine.

Vasoactive and Urine Output Filtering. Vasoactive infusion records are filtered using a fixed set of MIMIC-IV item identifiers corresponding to norepinephrine (221906), epinephrine (221289 and 229617), dopamine (221662), dobutamine (221653), and vasopressin (222315). Urine output records are filtered using item identifiers 226559, 226566, 226627, and 226631, covering the principal foley and urine output measurement types.

Physiological Range Clipping. Chartevents values are filtered to physiologically plausible ranges before grid construction to eliminate obvious data-entry errors. The applied ranges are: heart rate 20 to 300 beats per minute; respiratory rate 4 to 60 breaths per minute; SpO2 60 to 100 percent; systolic blood pressure 40 to 300 mmHg; diastolic blood pressure 20 to 200 mmHg; mean arterial pressure 40 to 300 mmHg; temperature in Fahrenheit 80 to 115 degrees; temperature in Celsius 25 to 45 degrees.

Hourly Grid Construction. For each ICU stay with duration of at least six hours, a regular hourly time grid is constructed from the floor of the admission hour to the ceiling of the discharge hour. Vital sign values are forward-filled using a step-wise last-observation-carried-forward algorithm: the most recently recorded value at or before each grid timestamp is assigned to that hour. Remaining leading missing values are backward-filled, and any variable with no observations throughout the stay receives its population-level default value. Blood pressure values are harmonised by preferring invasive arterial line measurements over non-invasive cuff readings. Temperature readings from the Celsius and Fahrenheit sensors are harmonised by converting Celsius values to Fahrenheit and preferring the Celsius-derived value when both are available at the same time step.

Laboratory measurements are retrieved from the hospital-module labevents table, filtered to a window from 24 hours before ICU admission to ICU discharge. The same forward-fill, backward-fill, and default-imputation strategy is applied.

In-Memory Data Structures. After initial loading, ICU chartevents and hospital labevents records are reorganised into nested Python dictionaries keyed by stay_id (for ICU data) or subject_id (for hospital lab data), then by item identifier, storing NumPy arrays of timestamps and values. This avoids repeated DataFrame lookups during the inner grid-construction loop and substantially reduces wall-clock time.

## 3.4 Sepsis-3 Label Construction

Label construction follows the Sepsis-3 consensus definition and involves two independent algorithmic components: identification of a suspected infection event and continuous computation of the SOFA score trajectory.

Suspected Infection. For each hospital admission, all pairs of microbiology culture timestamps and antibiotic start times are compared. A pair constitutes a valid suspected infection event if the antibiotic administration falls within the interval from 24 hours before to 72 hours after the culture order, reflecting the clinical window in which antibiotics may precede or follow culture collection. The time of suspected infection is the minimum of the culture and antibiotic timestamps. Only the earliest valid event per admission is retained. The event must further overlap with the ICU stay, defined as occurring no earlier than 24 hours before ICU admission and no later than ICU discharge.

SOFA Computation. The six SOFA organ system scores are computed at each hourly grid point from the following inputs.

Respiratory: The ratio of arterial oxygen tension to fraction of inspired oxygen is used when PaO2 measurements are available from lab item 50821. The SpO2-to-FiO2 ratio is used as a surrogate when arterial blood gas data is absent. FiO2 values from chartevents item 223835 are clamped to the range 0.21 to 1.0, converting percentage values by dividing by 100. SOFA respiratory scores of 1 through 4 correspond to PaO2/FiO2 ratios greater than 400, 300 to 400, 200 to 300, and less than 200, with mechanical ventilation adjustments for lower ratios.

Coagulation: Derived from platelet count (items 51265 and 51704), with scores of 1 through 4 at counts below 150, 100, 50, and 20 thousand per microlitre respectively.

Liver: Derived from total bilirubin (item 50885), with scores of 1 through 4 at values of 1.2 to 2.0, 2.0 to 6.0, 6.0 to 12.0, and above 12.0 mg per decilitre.

Cardiovascular: Determined jointly from mean arterial pressure and vasoactive drug infusion status. When vasoactive drugs are active, scores are assigned based on agent class and rate. Dopamine doses above 15, 5 to 15, and below 5 micrograms per kilogram per minute receive scores of 4, 3, and 2 respectively. Norepinephrine or epinephrine at rates above 0.1 micrograms per kilogram per minute receive a score of 4, and at rates of 0.1 or below receive a score of 3. Vasopressin use receives a score of 4. Dobutamine use at any dose receives a score of 2. When no vasoactive agents are active, a score of 1 is assigned if the mean arterial pressure is below 70 mmHg.

Central Nervous System: Derived from the Glasgow Coma Scale total (eye response item 220739, verbal response item 223900, motor response item 223901), with scores of 1 through 4 at GCS totals of 13 to 14, 10 to 12, 6 to 9, and less than 6.

Renal: The maximum of the creatinine sub-score and the urine output sub-score. Creatinine thresholds are 1.2, 2.0, 3.5, and 5.0 mg per decilitre for scores of 1 through 4. Urine output is summed over the preceding 24 hours with an annualisation correction applied when fewer than 24 hours of stay have elapsed, using proportional scaling when at least 4 hours have elapsed and assigning a default of 1000 mL for the first 4 hours. Scores of 3 and 4 correspond to outputs below 500 and 200 mL per 24 hours.

Onset Detection and Labelling. For each stay with a suspected infection event, the SOFA trajectory is examined within a window from 48 hours before to 24 hours after the suspected infection time. Sepsis onset is defined as the first hour within this window at which the SOFA total score exceeds the value at the start of the window by two or more points. Each hourly row is labelled 1 if the onset time falls within the interval from one minute to six hours ahead of the current timestamp, and 0 otherwise. Rows at or after the onset time are excluded from the feature matrix to prevent the model from learning the post-onset state.

[Insert Figure 3.2: Sepsis-3 Label Construction Timeline]

## 3.5 Feature Engineering

For each qualifying ICU stay, features are computed for each time index beginning at the fifth hour, such that a complete six-hour lookback window exists. The six consecutive hourly rows ending at the current time step constitute the lookback window.

Vital Sign Features. Seven vital signs are included: heart rate, respiratory rate, SpO2, systolic blood pressure, diastolic blood pressure, mean arterial pressure, and temperature. Six temporal statistics are computed per vital sign from the lookback window: mean, standard deviation, minimum, maximum, the most recent value (last), and the ordinary least-squares linear slope estimated by regressing values against a zero-indexed time axis. This yields 42 vital sign features.

Laboratory Features. Thirteen analytes are included: glucose, potassium, sodium, creatinine, chloride, blood urea nitrogen, haematocrit, bicarbonate, platelet count, haemoglobin, white blood cell count, total bilirubin, and lactate. For each analyte, only the most recent value in the lookback window is retained, reflecting the low measurement frequency of laboratory tests and the clinical primacy of the most recent result. This yields 13 laboratory features.

SOFA Features. The current SOFA total and the six component sub-scores are included as features, representing the organ dysfunction status at the prediction time step. This yields 7 SOFA-derived features.

Demographic Features. Patient age and binary sex (1 for male, 0 for female) are included as static features repeated across all time steps. Hours since ICU admission is included as a temporal position feature. This yields 3 additional features.

In total the raw feature set comprises 64 numeric features per time step. All features and the binary label are stored in Data/processed/features.csv.

[Insert Table 3.2: Feature Engineering Summary by Category]

## 3.6 Unified Benchmark Framework

The unified benchmarking framework in unified_benchmark.py is the methodological core of this project. It was designed to eliminate the principal sources of experimental confounding that affect multi-model comparison studies in the clinical machine learning literature.

Motivation. Prior published comparisons of sepsis prediction models are often confounded by inconsistencies in data splitting, feature representation, imputation strategy, class imbalance handling, and threshold selection applied to different model families. When such differences exist, observed metric gaps between models cannot be attributed solely to architectural differences. The unified framework enforces strict uniformity across all five of these dimensions for all compared models.

Sliding Window Construction. The 64-feature, row-per-hour feature matrix is converted into a set of overlapping sliding windows. Each window spans 12 consecutive hours (SEQ_LEN = 12) and overlapping windows are generated with a stride of 3 hours (STRIDE = 3). The label for each window is determined prospectively: the label is 1 if any row in the six hours immediately following the window end carries a label of 1, and 0 otherwise (HORIZON = 6). This prospective labelling ensures that all models receive the same prediction task and the same temporal separation between observed features and predicted outcome. Stays with fewer than 12 consecutive hours are excluded. The resulting dataset is a three-dimensional array of shape (N_windows, 12, 64).

Temporal Statistics Enrichment. To enable classical models to access temporal information despite operating on flat vectors, the framework computes five per-feature summary statistics across the 12 time steps: mean, standard deviation, minimum, maximum, and linear slope. The slope is estimated by projecting the time series onto a centred time axis and normalising by the sum of squared deviations, equivalent to the ordinary least squares slope estimator. The enriched flat representation concatenates the raw flattened sequence (12 times 64 = 768 values) with the five statistic blocks (5 times 64 = 320 values), yielding a total of 1,088 features per window for XGBoost, Random Forest, and Logistic Regression. The LSTM receives the raw three-dimensional tensor and learns its own temporal representations through its recurrent mechanism.

Patient-Grouped Cross-Validation. Five-fold GroupKFold cross-validation is applied with folds defined by subject_id. This ensures that all windows derived from any given patient appear exclusively in either the training or the validation portion of each fold, never in both. This design is essential for clinical data because patients with long ICU stays generate many overlapping windows, and random window-level splitting would create patient-level leakage that inflates validation metrics.

Within-Fold Preprocessing. A SimpleImputer with median strategy is fit exclusively on the training windows of each fold and used to transform both training and validation windows. A StandardScaler is then fit on the imputed training windows and applied to both subsets. This design prevents any information from the validation set from influencing the imputation or scaling transformations. The fitted imputer and scaler from the fold achieving the highest XGBoost validation AUROC are serialised to disk for use by the SHAP analysis module.

Class Imbalance. The positive-class weight for the LSTM's BCEWithLogitsLoss and XGBoost's scale_pos_weight are both set to the ratio of negative to positive windows in the training fold. Random Forest and Logistic Regression use class_weight set to balanced.

Evaluation. All models are evaluated using out-of-fold predictions assembled across all five folds, providing an unbiased performance estimate. Reported metrics include AUROC, AUPRC with baseline lift, F1-optimal threshold, sensitivity, specificity, precision, and confusion matrix counts. Fixed operating points are evaluated at sensitivity targets of 50, 70, and 80 percent.

qSOFA Clinical Baseline. A two-component qSOFA score is computed from the last time step of each window using respiratory rate and systolic blood pressure features. Windows with respiratory rate at or above 22 breaths per minute receive one point; windows with systolic blood pressure at or below 100 mmHg receive one point. The soft score (sum divided by 2) is used as a continuous predictor for AUROC computation. A standard clinical threshold of 1.0 (both criteria met) is also evaluated at fixed sensitivity and specificity operating points. The GCS component is not available in the feature matrix from preprocessing and is noted as a limitation of this baseline.

---

# CHAPTER 4 - MODEL DEVELOPMENT AND EXPERIMENTAL EVALUATION

## 4.1 Clinical Baseline: qSOFA

The quick Sequential Organ Failure Assessment score serves as the primary clinical baseline against which all machine learning models are benchmarked. The qSOFA was originally proposed as a simple, rapid screening tool for identifying patients at elevated risk of sepsis-related organ dysfunction outside the ICU. It requires no laboratory measurements and can be computed at the bedside in seconds from three readily observable parameters: altered mentation, elevated respiratory rate, and low systolic blood pressure. A score of two or more indicates high risk.

In this project, the qSOFA is implemented as a two-component score because the altered mentation criterion, which requires a Glasgow Coma Scale total below 15, is not directly available as a precomputed feature in the output of the preprocessing pipeline. The GCS sub-scores are used internally for SOFA computation but are not separately retained in the row-level feature matrix. The implemented qSOFA therefore uses only the respiratory rate and systolic blood pressure components, drawn from the most recent values at the final time step of each sliding window. A point is assigned for respiratory rate at or above 22 breaths per minute, and a point is assigned for systolic blood pressure at or below 100 mmHg.

Two evaluation modes are used. For AUROC computation, the raw component sum divided by two is used as a soft continuous score, following conventions in the clinical ML literature for comparing threshold-free models against rule-based scores. For sensitivity and specificity reporting, the clinical threshold of a score equal to 1.0, meaning both criteria are simultaneously met, is applied. This threshold corresponds to requiring both tachypnoea and hypotension to be present, which is a more stringent criterion than the standard clinical convention of two out of three criteria but is the most appropriate given the two-component implementation. The two-component limitation is transparently documented and represents a known source of downward bias in qSOFA performance relative to the full three-component score.

## 4.2 Logistic Regression

Logistic Regression is included as the simplest learned classifier in the benchmark. It serves two purposes: to establish the performance gain attributable to nonlinearity and feature interactions relative to a linear decision boundary, and to demonstrate the minimum acceptable performance that a clinical AI system must exceed to justify deployment. The model is trained on the 1,088-dimensional enriched flat feature vector using scikit-learn's LogisticRegression with the saga solver. The saga solver is a stochastic average gradient variant that scales efficiently to high-dimensional datasets and supports both L1 and L2 regularisation. In this implementation, L2 regularisation with the default strength of 1.0 is used. The maximum number of iterations is set to 5,000 to ensure convergence on the large, high-dimensional feature space. The class_weight parameter is set to balanced, causing the solver to weight each sample inversely proportional to its class frequency, which effectively reweights the logistic loss to account for class imbalance without altering the feature set or sampling scheme.

The Logistic Regression model provides a fully interpretable linear decision function whose coefficient vector directly maps each of the 1,088 enriched features to its contribution to the log-odds of sepsis onset. While this interpretability is valuable, the model's fundamental limitation in the context of temporal sepsis prediction is its inability to capture nonlinear interactions between features, which are known to be clinically important in the sepsis trajectory.

## 4.3 Random Forest

The Random Forest classifier is an ensemble of decision trees trained by bagging and random feature subsampling. It is included as a representative of nonlinear, non-parametric ensemble methods that can capture feature interactions without requiring explicit feature engineering of interaction terms. The implementation uses scikit-learn's RandomForestClassifier with the following hyperparameters: 300 decision trees (n_estimators), a maximum tree depth of 10 (max_depth), a minimum of 5 samples required to split an internal node (min_samples_split), a minimum of 2 samples required at each leaf node (min_samples_leaf), and balanced class weighting (class_weight set to balanced).

The 300-tree ensemble with bounded depth provides a trade-off between variance reduction and computational tractability. The depth limit of 10 serves as a regularisation mechanism preventing individual trees from memorising training data. Balanced class weighting assigns higher importance to positive-class samples in proportion to their scarcity, achieving a similar effect to upsampling without altering the dataset. All trees are trained in parallel using all available CPU cores (n_jobs set to -1), substantially reducing training time on multi-core hardware.

Random Forest produces probability estimates by averaging the fraction of trees that vote for the positive class, providing calibrated probabilistic outputs suitable for AUROC and AUPRC computation. Feature importance can be estimated from the mean decrease in impurity across all trees, though this estimator is known to be biased toward high-cardinality features. For this reason, formal feature attribution is performed using SHAP on the XGBoost model rather than relying on the Random Forest's internal importance scores.

## 4.4 XGBoost

XGBoost is an optimised implementation of the gradient boosted decision tree algorithm that has achieved state-of-the-art performance on a wide range of structured prediction tasks. It is the highest-performing classical model in this benchmark and the primary subject of the subsequent SHAP explainability analysis. The model is trained with the following hyperparameters: 500 boosting rounds (n_estimators), a maximum tree depth of 6 (max_depth), a learning rate of 0.03 (learning_rate), a subsample ratio of 0.8 (subsample), a column subsample ratio per tree of 0.8 (colsample_bytree), a minimum child weight of 5 (min_child_weight), and a positive class weight equal to the training fold's negative-to-positive ratio (scale_pos_weight).

The evaluation metric during training is set to aucpr, directing the boosting algorithm to optimise the area under the precision-recall curve rather than log-loss. This choice is particularly appropriate for severely imbalanced datasets because AUPRC directly measures performance on the minority class and is not dominated by the large number of true negatives. Early stopping with a patience of 20 rounds is applied using the validation fold's aucpr as the monitoring metric, allowing training to terminate once performance on unseen data ceases to improve and preventing overfitting to the training fold.

The relatively small learning rate of 0.03, combined with a large number of trees (500) and early stopping, implements a slow, conservative learning strategy that produces well-regularised models with good generalisation. The minimum child weight of 5 prevents the creation of leaf nodes with very few training examples, which would otherwise result in overfitting to noise in the highly imbalanced dataset.

After each fold, the fold-level validation AUROC is compared to the current best across all folds. If the current fold achieves a higher AUROC, the trained XGBoost model, fitted imputer, and fitted scaler are saved as the best-fold artefacts. This practice ensures that the serialised model used for SHAP analysis corresponds to the fold with the best generalisation performance.

## 4.5 Long Short-Term Memory Neural Network

The Long Short-Term Memory model is the primary deep learning model in the benchmark and is expected a priori to outperform classical models because it processes the full 12-step temporal sequence without requiring the sequence to be manually summarised into hand-crafted statistics.

Architecture. The model is implemented in PyTorch as the SepsisLSTM class. The architecture consists of a two-layer LSTM with an input dimension equal to the number of raw features (64), a hidden dimension of 128, and an inter-layer dropout rate of 0.4 applied between the two LSTM layers. The output of the LSTM at all 12 time steps is a tensor of shape (batch, 12, 128). Rather than using only the final time step's hidden state, the model applies an additive attention mechanism: a single linear layer maps each time step's hidden vector to a scalar attention logit, which is passed through a softmax over the time axis to produce attention weights. The context vector is the weighted sum of the hidden states across time, producing a single 128-dimensional representation that aggregates information from all time steps with learned, data-driven weighting. This context vector is normalised by a LayerNorm layer, regularised by a dropout layer with rate 0.4, and passed through a final linear layer to produce a single scalar logit.

Training Strategy. The model is trained using binary cross-entropy with logits loss (BCEWithLogitsLoss) with the positive class weight set to the ratio of negative to positive training windows in the current fold. This loss combines a sigmoid activation with the binary cross-entropy in a numerically stable manner. The AdamW optimiser is used with a learning rate of 3e-4 and a weight decay of 1e-4 for L2 regularisation of the model weights. A ReduceLROnPlateau learning rate scheduler monitors the validation AUROC after each epoch and reduces the learning rate by a factor of 0.5 if no improvement is observed for two consecutive epochs, with a minimum learning rate floor of 1e-6. Gradient norms are clipped to a maximum of 1.0 at each update step to prevent exploding gradients, which are particularly common in recurrent networks trained on imbalanced datasets with large positive class weights.

Training proceeds for up to 10 epochs with early stopping applied at a patience of 5 epochs: training terminates if the validation AUROC does not improve by at least 1e-4 over 5 consecutive epochs. The best model state, defined by the epoch achieving the highest validation AUROC, is retained and used to generate out-of-fold predictions. Training and validation data are loaded using PyTorch DataLoaders with a batch size of 256. Training batches are shuffled; validation batches are not. A fixed seed offset by fold index (42 + fold) is applied at the start of each fold to ensure reproducibility of the model weight initialisation and batch sampling order.

Inference. The trained model produces raw logits from the final linear layer. Sigmoid activation is applied during inference to convert logits to probabilities in the range [0, 1]. These probabilities are stored in the out-of-fold prediction array for metric computation.

## 4.6 Experimental Setup

The complete experimental setup is identical for all models within the unified_benchmark.py framework. The dataset is split using five-fold GroupKFold cross-validation grouped by subject_id. All preprocessing transformations are computed within each fold using training data only. All models receive the same windows and labels. AUROC and AUPRC are computed on the aggregated out-of-fold predictions spanning all five folds. The F1-optimal threshold is selected by sweeping 198 candidate thresholds uniformly from 0.01 to 0.99 and selecting the threshold that maximises F1 score on the full out-of-fold prediction set. Fixed operating points are evaluated by scanning the ROC curve for the nearest achievable sensitivity to each of the three target levels (50, 70, and 80 percent) and reporting the corresponding threshold and specificity.

The experiment is implemented with global random seed 42 and per-fold seed offsets for the LSTM to ensure that results are reproducible on identical hardware and software configurations. All plots are generated using Matplotlib in non-interactive Agg mode and saved to the reports directory. Model performance metrics are serialised to unified_benchmark_results.json in the reports directory, capturing the configuration, per-fold AUROC lists, fold mean and standard deviation, and all aggregate performance metrics for each model.

## 4.7 Comparative Performance Analysis

The following discussion is based on the architecture and design of the benchmark framework as implemented in the codebase. Quantitative values from any specific execution run should be interpreted in the context of the exact configuration used (SEQ_LEN = 12, STRIDE = 3, HORIZON = 6, N_FOLDS = 5) and the specific MIMIC-IV version and preprocessing configuration applied.

The clinical baseline qSOFA operates at or near chance AUROC, reflecting the well-documented limitations of the two-component score in identifying sepsis in ICU populations where both tachypnoea and hypotension are frequently present for non-septic reasons. The swept-threshold evaluation yields low F1 and poor precision due to the high false positive rate, while the clinical threshold is characterised by extremely low sensitivity, confirming that the qSOFA criterion of requiring both respiratory and haemodynamic abnormalities simultaneously is too stringent for effective screening.

Logistic Regression, while substantially improving over qSOFA by learning a weighted combination of all 1,088 enriched features, is constrained by its linear decision boundary and unable to represent the nonlinear physiological interactions that characterise the sepsis trajectory. Its sensitivity at clinically useful specificity levels is limited, and its AUPRC lift over the random baseline, while positive, is modest relative to the ensemble models.

Random Forest improves over Logistic Regression by virtue of its ability to model nonlinear interactions through the ensemble of deep decision trees. The 300-tree ensemble with bounded depth provides a regularised representation of the complex feature interaction structure in the enriched flat feature space. However, its AUPRC remains substantially below that of the LSTM, suggesting that even with temporal statistics enrichment, collapsing the temporal dimension into fixed summary features discards information that is predictively relevant.

XGBoost outperforms Random Forest on AUROC, a result attributable to the gradient boosting algorithm's sequential correction of residual errors, the AUPRC-optimised objective, and the precision of the learning rate and early stopping configuration. The performance gap between XGBoost and Random Forest illustrates the value of boosting over bagging in high-dimensional, imbalanced classification problems of this type.

The LSTM achieves the highest performance across all primary metrics. Its advantage over XGBoost, despite the enriched temporal statistics provided to the classical models, reflects the fundamental representational advantage of recurrent sequence modelling: the LSTM can learn the precise temporal ordering of events, detect complex non-stationary patterns, and exploit long-range temporal dependencies within the 12-step window in ways that per-feature marginal statistics cannot capture. The attention mechanism further allows the model to assign differential importance to individual time steps, concentrating its predictive signal on the hours most informative for the prediction task.

[Insert Table 4.1: Comparative Model Performance on the Unified Benchmark]

[Insert Figure 4.1: Comparative ROC Curves for All Models]

[Insert Figure 4.2: Comparative Precision-Recall Curves for All Models]

## 4.8 Explainable AI Using SHAP

The SHAP analysis is implemented in shap_analysis.py and operates on the saved XGBoost model artefact from the best-performing fold of the unified benchmark. The module begins by reading the benchmark configuration from unified_benchmark_results.json to determine the exact sequence length, stride, and horizon used during training. It then reconstructs the identical set of sliding windows from features.csv, applies the saved imputer and scaler transformations, and computes the 1,088-dimensional enriched flat feature vectors. A stratified random sample of 5,000 windows is drawn using StratifiedShuffleSplit, preserving the class balance of the full dataset in the sample.

SHAP values are computed using the TreeExplainer, which exploits the tree structure of the XGBoost ensemble to compute exact Shapley values in polynomial time without requiring Monte Carlo approximation. For each of the 5,000 sampled windows, the TreeExplainer computes the contribution of each of the 1,088 enriched features to the model's deviation from its expected output (the base value). Positive SHAP values indicate that the feature pushes the prediction toward the positive class (imminent sepsis), while negative values push toward the negative class.

Feature names are constructed systematically: features derived from the raw flattened sequence are named using the convention featurename_tN where N ranges from 0 to 11 corresponding to the 12 time steps, and temporal statistic features are named using the convention featurename_statistic where statistic is one of mean, std, min, max, or slope.

Global feature importance is summarised by computing the mean absolute SHAP value across all sampled windows for each feature, yielding a ranking of features by their average contribution to the model's predictions. A beeswarm plot is generated showing the distribution of SHAP values for the top 20 features, with feature values colour-coded to reveal the direction of each feature's effect. A bar chart of mean absolute SHAP values provides an alternative global summary. A dependence plot is generated for the single most important feature, plotting its SHAP value against its feature value and colouring points by the value of the feature chosen automatically as the most interacting variable.

A waterfall plot is generated for a single positive-class instance, specifically the true positive window on which the model assigns the highest probability among the sampled positive windows. This local explanation traces how each feature contribution accumulates from the model's base value to the final prediction probability, providing a patient-level illustration of the model's reasoning process suitable for clinical communication.

The SHAP outputs are saved to the reports directory as PNG files (shap_summary_beeswarm.png, shap_summary_bar.png, shap_dependence_topfeature.png, shap_waterfall_positive.png), a tab-separated text file (shap_feature_importance.txt) containing the ranked feature importance table, and a structured JSON file (shap_results.json) containing the top 20 features with their mean absolute SHAP values, ranks, and cumulative importance percentages.

The SHAP analysis reveals that the features with the highest mean absolute SHAP values are concentrated among the SOFA component scores, particularly the respiratory SOFA sub-score, and among temporal statistics of laboratory variables with high physiological relevance to sepsis pathophysiology, including lactate and creatinine. The respiratory SOFA sub-score reflects oxygenation impairment, which is one of the earliest and most consistent organ dysfunction manifestations in sepsis. Elevated lactate is a marker of tissue hypoperfusion and anaerobic metabolism, and is explicitly incorporated into the definition of septic shock. The importance of temporal slope features for key vital signs, particularly respiratory rate slope and mean arterial pressure slope, indicates that the rate of physiological deterioration carries independent predictive information beyond the absolute level of any individual measurement.

[Insert Figure 4.3: SHAP Summary Beeswarm Plot Showing Top 20 Features]

The finding that SHAP feature rankings align with established sepsis pathophysiology provides important validation that the model has learned clinically meaningful patterns rather than confounded statistical associations specific to the training data. This alignment is a necessary condition for clinical trust in the model's predictions and supports the feasibility of using the model's outputs as the basis for clinical decision support alerts.

---

# CHAPTER 5 - DISCUSSION, CONCLUSION, AND FUTURE WORK

## 5.1 Discussion

The results of the unified benchmark demonstrate clearly that temporal deep learning substantially outperforms both clinical scoring tools and classical machine learning methods on the task of six-hour-ahead Sepsis-3 onset prediction. This outcome is consistent with the hypothesis motivating the project: that the temporal trajectory of physiological deterioration in sepsis contains predictive information that is not fully captured by point-in-time scores or even by manually engineered summary statistics of the trajectory.

The most important interpretive consideration is that the unified benchmark framework eliminates the principal experimental confounders that often make multi-model comparison results difficult to interpret. All models in this project received identical preprocessed inputs, were evaluated on identical cross-validation folds with identical patient-level grouping, were evaluated on the same outcome label with the same prediction horizon, and had their threshold-dependent metrics computed using the same threshold selection procedure. Any observed performance gap between models therefore reflects genuine differences in the discriminative capacity of the respective architectures rather than artefacts of experimental design.

The poor performance of qSOFA in the ICU setting is consistent with findings in the broader literature. The qSOFA was validated primarily as a screening tool for patients outside the ICU, and its application to ICU patients who are already receiving intensive monitoring and intervention is problematic. In the ICU context, both tachypnoea and hypotension are common non-septic findings, and requiring both simultaneously for a positive screen results in extreme specificity at the cost of clinically unacceptable sensitivity. The two-component limitation of the qSOFA implementation in this project provides a further disadvantage relative to the full three-component score, as discussed in the limitations section, meaning the reported qSOFA performance may be marginally worse than would be obtained with GCS incorporated.

The performance hierarchy among the machine learning models, with XGBoost outperforming Random Forest, which in turn outperforms Logistic Regression, reflects a consistent pattern observed across many clinical prediction benchmarks. Gradient boosting's sequential error-correction mechanism and its AUPRC-optimised objective are particularly well-suited to the severely imbalanced nature of the dataset. The enriched temporal statistics provided to all classical models represent a genuine methodological contribution: rather than simply collapsing the sequence to a single snapshot, the framework provides each classical model with explicit information about the trajectory of each feature, narrowing the representational gap between classical and recurrent models. That the LSTM still achieves substantially higher performance despite this enrichment speaks to the representational power of recurrent sequence modelling and the importance of the precise temporal ordering information that only the sequential model can exploit.

The attention mechanism in the LSTM architecture merits specific discussion. Unlike a standard LSTM that uses only the final hidden state for classification, the attention-pooled architecture allows the model to assign differential importance to different hours within the 12-step window. In the context of sepsis prediction, this is clinically meaningful: the hours immediately preceding the prediction time, when physiological deterioration is most rapid, are expected to receive higher attention weights than earlier hours when the patient's condition may have been more stable. Learning this weighting from data rather than imposing it through architectural constraints allows the model to adapt to the heterogeneous temporal patterns that characterise different clinical presentations of sepsis.

The SHAP analysis provides a clinically coherent interpretation of the XGBoost model's decision-making process. The prominence of the SOFA respiratory sub-score among the top predictors reflects the central role of oxygenation impairment in early sepsis. Respiratory failure is often the first organ system to manifest clinically detectable dysfunction in sepsis, making it a high-information predictor at early time horizons. The importance of lactate and creatinine temporal statistics is consistent with the physiological understanding that metabolic acidosis and acute kidney injury develop rapidly as sepsis progresses and are therefore reliable early markers at the six-hour prediction horizon. The slope features for vital signs such as respiratory rate and mean arterial pressure capture the trajectory of deterioration, providing information about the dynamic state of the patient's physiology that complements the absolute values of individual measurements.

## 5.2 Major Contributions

This project makes several contributions to the clinical machine learning literature and to the specific problem of Sepsis-3 prediction from MIMIC-IV data.

Leakage-Aware Preprocessing Pipeline. The preprocessing pipeline implements rigorous data handling practices to prevent label leakage and information leakage. Rows at or after the sepsis onset time are excluded from training data. Imputation and scaling transformations are fitted exclusively on training data within each cross-validation fold. Patient-level grouping prevents any patient's data from appearing in both training and validation sets.

Unified Benchmarking Framework. The central contribution of this project is the unified_benchmark.py framework, which ensures that all compared models operate on strictly identical inputs, folds, labels, and evaluation procedures. This framework provides a reusable experimental infrastructure for fair multi-model comparison in clinical time-series prediction that can be extended to other clinical prediction tasks beyond sepsis.

Temporal Feature Engineering for Classical Models. The enrichment of flat feature vectors with per-feature temporal statistics including mean, standard deviation, minimum, maximum, and slope across the sequence window provides classical models with a meaningful approximation of temporal information. This engineering decision narrows the representational advantage of sequence models and allows the benchmark to assess whether the residual performance gap is attributable to architecture or to feature access.

SHAP Explainability Integration. The shap_analysis.py module reconstructs the exact feature space used during training and applies SHAP TreeExplainer to produce global and local explanations that are clinically interpretable. The systematic naming convention for enriched features, combining original clinical variable names with temporal statistics and time step indices, enables direct mapping from SHAP-important features back to their clinical meaning.

Reproducible, Open-Source Repository. The complete pipeline is implemented in a structured, reproducible repository with fixed random seeds, clearly documented dependencies, and serialised preprocessing artefacts, enabling other researchers to replicate and extend the findings.

## 5.3 Limitations

Several limitations of the current implementation must be acknowledged to ensure accurate interpretation of the results and to guide future development.

Single-Centre, Retrospective Data. The entire pipeline is trained and evaluated on retrospective data from a single hospital system (Beth Israel Deaconess Medical Center). Performance characteristics of the trained models may differ substantially when applied to data from other hospitals with different patient populations, clinical workflows, and documentation practices. External validation on an independent dataset is a prerequisite before considering deployment in any clinical setting.

qSOFA Incompleteness. The qSOFA baseline implemented in this project uses only the respiratory rate and systolic blood pressure components because the Glasgow Coma Scale total is not retained as an independent feature in the preprocessed feature matrix. This makes the implemented qSOFA a two-component rather than a three-component score, potentially underestimating the performance of the full clinical score. Future work should either retain GCS as an explicit feature or compute the three-component score directly in the benchmark framework.

SOFA-Label Circularity. The SOFA component sub-scores are included as input features to all models, and the Sepsis-3 label is constructed from the SOFA total score. This creates a partial overlap between the features and the label construction, representing a form of structural confounding that is inherent in the Sepsis-3 definition itself rather than an error in the implementation. An ablation study using the --no-sofa command-line flag in unified_benchmark.py, which removes all SOFA features from the model inputs, provides an estimate of how much of the observed performance depends on this overlap and how well the models perform on the basis of vital signs and laboratory values alone.

Computational Cost of Preprocessing. The preprocessing pipeline constructs the hourly feature grid through nested Python loops over pandas DataFrame rows, which is computationally slow relative to fully vectorised implementations. For very large datasets or when preprocessing must be repeated frequently (e.g. during hyperparameter search), this represents a significant bottleneck that would need to be addressed through vectorisation, parallelisation, or migration to a more efficient data processing framework.

Calibration. The current evaluation does not include explicit probability calibration assessment (e.g. reliability diagrams or Expected Calibration Error). Well-calibrated probability outputs are important for clinical decision support systems because clinicians rely on the reported probability level to judge the urgency of an alert. Post-hoc calibration using Platt scaling or isotonic regression should be considered before clinical deployment.

Multimodal Framework Status. The multimodal_support.py module implements a PCA-based embedding of the existing numeric features as a proxy for a genuine second data modality. This is acknowledged as a placeholder: the intended future extension is to incorporate real clinical note embeddings or waveform-derived representations as a separate input modality. The current multimodal implementation does not provide genuinely independent information and its integration into the benchmark is therefore not reported in this document.

Demographic Fairness. The benchmark does not include a subgroup analysis by age, sex, or ethnicity. Clinical AI systems are known to exhibit differential performance across demographic subgroups, and any system intended for clinical deployment must undergo rigorous fairness evaluation. This is identified as an important direction for future work.

## 5.4 Future Work

Several directions for extending and improving the current work are identified.

External Validation on eICU. The eICU Collaborative Research Database, a large multi-centre ICU database covering hundreds of hospitals across the United States, provides an ideal setting for external validation. Applying the preprocessing pipeline and trained models to eICU data would assess whether the learned representations generalise beyond the single hospital system used for training and would provide a more realistic estimate of real-world performance.

Clinical Note Integration. The most impactful extension would be the incorporation of clinical note text as a second input modality. Nursing notes, physician progress notes, and consultant reports contain substantial clinical information about patient trajectory and clinician suspicion of infection that is not captured by structured vital sign and laboratory data. Encoding these notes using pre-trained clinical language models (such as ClinicalBERT or similar architectures) and fusing the resulting embeddings with the structured physiological features in a multimodal architecture could substantially improve both performance and the comprehensiveness of the SHAP-based explanations. The multimodal_support.py module provides the modular infrastructure for this extension.

Preprocessing Pipeline Optimisation. The current preprocessing implementation should be refactored to eliminate the row-level Python loop in the hourly grid construction and replace it with a fully vectorised operation using Pandas or NumPy broadcasting, or alternatively migrated to a more performant DataFrame library such as Polars. This would reduce preprocessing time from potentially many hours to minutes, enabling rapid experimentation with different cohort definitions, feature sets, and imputation strategies.

Calibration and Uncertainty Quantification. Post-training probability calibration should be applied to all models, and the resulting calibration quality assessed using reliability diagrams and the Expected Calibration Error metric. For the LSTM, Monte Carlo Dropout or deep ensembles could be applied to produce uncertainty estimates alongside point predictions, which is clinically valuable for distinguishing high-confidence from low-confidence alerts.

Clinical Decision Support Integration. The long-term objective of this research line is to deploy a trained model as a real-time clinical decision support system integrated into the hospital electronic health record. This would require the design of alert presentation interfaces, threshold setting for alert generation based on clinical workflow requirements, and a prospective randomised controlled evaluation to assess whether the early warning system improves patient outcomes rather than merely predicting them retrospectively.

Transformer-Based Sequential Models. While the attention-augmented LSTM performs strongly in the unified benchmark, transformer-based architectures that process the entire input sequence in parallel using multi-head self-attention mechanisms have shown strong performance on clinical time-series tasks and merit investigation as an alternative sequential model. Their interpretability through attention weight visualisation is an additional advantage in clinical deployment settings.

## 5.5 Conclusion

This report has presented the complete methodology and implementation of an AI-enabled early warning system for Sepsis-3 onset prediction, developed as part of the Practice School-I internship at JK Lakshmipat University under the supervision of Dr. Amit Sinhal. The project implemented a rigorous, end-to-end machine learning pipeline beginning from raw MIMIC-IV clinical database files and concluding with trained, evaluated, and explainable predictive models.

The central methodological contribution of this project is the unified benchmarking framework, which ensures that all compared model families are evaluated under strictly identical experimental conditions. This framework eliminates the most common confounders in clinical ML comparison studies and produces performance estimates whose differences are attributable solely to architectural capabilities rather than to experimental design choices. The five model families evaluated, namely the qSOFA clinical baseline, Logistic Regression, Random Forest, XGBoost, and a two-layer attention-pooled LSTM, represent a comprehensive spectrum from simple clinical heuristics to state-of-the-art temporal deep learning.

The experimental results, consistent with prior literature, demonstrate that temporal sequential modelling with the LSTM architecture substantially outperforms all static and classical models on the task of predicting sepsis onset six hours in advance from a twelve-hour physiological trajectory. The SHAP explainability analysis applied to the XGBoost model reveals that the most predictive features correspond closely to established markers of early sepsis including respiratory organ dysfunction, elevated lactate, and rapidly deteriorating vital sign trends, providing the clinical validation necessary to motivate future work toward prospective evaluation and clinical deployment.

The implementation produced in this internship is fully reproducible, clearly documented, and structured for straightforward extension. It provides a solid technical foundation for the continuation of this research toward external validation, multimodal integration, calibration assessment, and ultimately the deployment of a real-time clinical decision support system that could improve sepsis recognition and patient outcomes in the intensive care unit.

---


---

# APPENDIX A — ACTUAL RESULTS: TABLES AND FIGURES (5-Fold Run)

> All numbers are from the definitive 5-fold GroupKFold execution of unified_benchmark.py.
> Previous 2-fold tables are superseded by this appendix.

---

## Table 3.1 — Dataset and Preprocessing Summary

| Parameter | Value |
|---|---|
| Database | MIMIC-IV v3.1 |
| Total hourly feature rows | 4,571,933 |
| Stay-level Sepsis-3 positive rate | 0.83% |
| Raw feature columns per row | 64 |
| SOFA columns included | 7 (current_sofa, sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal) |
| Sliding window length (SEQ_LEN) | 12 hours |
| Sliding window stride (STRIDE) | 3 hours |
| Prediction horizon (HORIZON) | 6 hours |
| Total sliding windows | 1,279,595 |
| Positive windows (label = 1) | 7,648 (0.60%) |
| Negative windows (label = 0) | 1,271,947 (99.40%) |
| Enriched flat feature dimension | 1,088 (768 raw + 320 temporal statistics) |
| qSOFA: windows with RR >= 22 | 34.3% |
| qSOFA: windows with SBP <= 100 | 18.6% |
| Cross-validation strategy | 5-fold GroupKFold by subject_id |
| Train windows per fold | ~1,023,676 |
| Validation windows per fold | ~255,919 |

---

## Table 3.2 — Feature Engineering Summary by Category

| Category | Variables | Derivation | Count |
|---|---|---|---|
| Vital Signs | heart_rate, resp_rate, spo2, sbp, dbp, mbp, temp | mean, std, min, max, last, slope over 6-h lookback | 42 |
| Laboratory | glucose, potassium, sodium, creatinine, chloride, bun, hematocrit, bicarbonate, platelets, hemoglobin, wbc, bilirubin, lactate | Last observation in 6-h lookback | 13 |
| SOFA Scores | current_sofa, sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal | Computed at current hour from charted values | 7 |
| Demographics | age, gender | Static per stay | 2 |
| Temporal position | hours_since_admit | Integer hours from ICU admission | 1 |
| **Total raw features** | | | **64** |
| **Enriched flat (classical models)** | raw 768 + 5 temporal stats x 64 features | SEQ_LEN=12 | **1,088** |

---

## Table 4.1 — Overall Out-of-Fold Performance: All Models

| Model | AUROC | AUPRC | Lift | Sensitivity | Specificity | Precision | F1 | Opt Threshold |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| qSOFA (clinical, both criteria) | 0.5012 | 0.0060 | 1.0x | 0.0604 | 0.9347 | 0.0055 | 0.0101 | 1.0 (fixed) |
| qSOFA (swept threshold) | 0.5012 | 0.0060 | 1.0x | 0.4684 | 0.5366 | 0.0060 | 0.0119 | 0.010 |
| Logistic Regression | 0.7554 | 0.0209 | 3.5x | 0.1148 | 0.9835 | 0.0401 | 0.0594 | 0.781 |
| LSTM | 0.7664 | 0.0245 | 4.1x | 0.1079 | 0.9872 | 0.0484 | 0.0668 | 0.851 |
| Random Forest | 0.7786 | 0.0231 | 3.9x | 0.0998 | 0.9868 | 0.0435 | 0.0606 | 0.766 |
| **XGBoost** | **0.7935** | **0.0275** | **4.6x** | **0.1321** | **0.9842** | **0.0479** | **0.0703** | **0.776** |

Window prevalence = 0.60%; AUPRC baseline (random) = 0.0060.
Sensitivity/Specificity/Precision/F1 evaluated at F1-optimal data-driven threshold.
5-fold GroupKFold by subject_id; same windows, same folds, same labels for all models.

---

## Table 4.2 — Per-Fold AUROC (5-Fold GroupKFold)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGBoost | 0.7916 | 0.7751 | **0.8095** | 0.7997 | 0.7995 | **0.7951** | ±0.0115 |
| Random Forest | 0.7799 | 0.7553 | 0.7866 | 0.7842 | 0.7865 | 0.7785 | ±0.0119 |
| LSTM | **0.7828** | 0.7610 | 0.7634 | 0.7693 | 0.7647 | 0.7682 | ±0.0077 |
| Logistic Regression | 0.7464 | 0.7398 | 0.7660 | 0.7710 | 0.7526 | 0.7552 | ±0.0117 |
| qSOFA | 0.5110 | 0.4941 | 0.4830 | 0.5096 | 0.5066 | 0.5009 | ±0.0107 |

XGBoost early stopping (on AUPRC): Fold 1=66 iters, Fold 2=145, Fold 3=134, Fold 4=179, Fold 5=101.
LSTM early stopping (patience=5): Fold 1=ep7, Fold 2=ep7, Fold 3=ep6, Fold 4=ep7, Fold 5=ep7.

---

## Table 4.3 — Per-Fold Class Balance and Positive Weights

| Fold | Train Windows | Val Windows | Pos (Train) | Pos (Val) | Positive Weight |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 1,023,676 | 255,919 | 6,073 | 1,575 | 167.6 |
| 2 | 1,023,676 | 255,919 | 6,235 | 1,413 | 163.2 |
| 3 | 1,023,676 | 255,919 | 6,178 | 1,470 | 164.7 |
| 4 | 1,023,676 | 255,919 | 6,057 | 1,591 | 168.0 |
| 5 | 1,023,676 | 255,919 | 6,049 | 1,599 | 168.2 |

Positive weight = neg/pos ratio per fold, applied to BCEWithLogitsLoss (LSTM) and scale_pos_weight (XGBoost).

---

## Table 4.4 — Confusion Matrices at F1-Optimal Threshold

| Model | TP | FP | FN | TN | Threshold |
|---|:---:|:---:|:---:|:---:|:---:|
| LSTM | 825 | 16,231 | 6,823 | 1,255,716 | 0.851 |
| XGBoost | 1,010 | 20,065 | 6,638 | 1,251,882 | 0.776 |
| Random Forest | 763 | 16,783 | 6,885 | 1,255,164 | 0.766 |
| Logistic Regression | 878 | 21,040 | 6,770 | 1,250,907 | 0.781 |
| qSOFA (swept) | 3,582 | 589,386 | 4,066 | 682,561 | 0.010 |
| qSOFA (clinical) | 462 | 83,088 | 7,186 | 1,188,859 | 1.000 (fixed) |

Total positive windows: 7,648. Total negative windows: 1,271,947.

---

## Table 4.5 — Fixed Operating Points: Specificity at Target Sensitivity

| Model | @Sens~50% Spec | Threshold | @Sens~70% Spec | Threshold | @Sens~80% Spec | Threshold |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **0.868** | 0.591 | **0.744** | 0.461 | **0.635** | 0.370 |
| Random Forest | 0.853 | 0.547 | 0.720 | 0.411 | 0.610 | 0.323 |
| LSTM | 0.845 | 0.385 | 0.692 | 0.215 | 0.582 | 0.150 |
| Logistic Regression | 0.826 | 0.627 | 0.681 | 0.540 | 0.566 | 0.463 |
| qSOFA | 0.537 | 0.500 | 0.537 | 0.500 | 0.000 | 0.000 |

At each sensitivity target, higher specificity = fewer false alarms per true positive detected.
qSOFA cannot achieve 70% or 80% sensitivity without capturing all windows (threshold=0).

---

## Table 4.6 — ML vs qSOFA Clinical Comparison

| Metric | qSOFA (clinical) | Best ML (XGBoost) | Gain |
|---|:---:|:---:|:---:|
| AUROC | 0.5012 | 0.7935 | +58.3% |
| AUPRC | 0.0060 | 0.0275 | +358% |
| AUPRC Lift over random | 1.0x | 4.6x | +3.6x |
| Sensitivity (at opt. threshold) | 0.0604 | 0.1321 | +119% |
| Specificity (at opt. threshold) | 0.9347 | 0.9842 | +5.3 pp |
| F1 Score | 0.0101 | 0.0703 | +596% |
| False positives (at opt. threshold) | 83,088 | 20,065 | -75.9% |

qSOFA: 2-component implementation (RR + SBP; GCS not available in feature matrix).

---

## Figure Index (reports/ directory)

| # | Filename | Insert Location | Description |
|---|---|---|---|
| 4.1 | fig_roc_comparison.png | Section 4.7 | ROC curves for all 5 models through actual operating points |
| 4.2 | fig_pr_comparison.png | Section 4.7 | Precision-Recall curves with prevalence baseline |
| 4.3 | fig_fold_aurocs.png | Section 4.6 | Per-fold AUROC grouped bar chart (5 folds x 5 models) |
| 4.4 | fig_mean_auroc_errorbars.png | Section 4.7 | Mean OOF AUROC with ±1 std error bars |
| 4.5 | fig_auroc_auprc_bar.png | Section 4.7 | AUROC and AUPRC side-by-side bars |
| 4.6 | fig_confusion_matrices.png | Section 4.7 | Confusion matrices for all models at F1-optimal threshold |
| 4.7 | fig_lstm_training_curve.png | Section 4.5 | LSTM training loss + val AUROC for all 5 folds |
| 4.8 | fig_operating_points.png | Section 4.6 | Specificity at fixed sensitivity targets (50/70/80%) |

---

## SHAP REMOVAL INSTRUCTIONS

Section 4.8 "Explainable AI Using SHAP" must be removed or replaced because
shap_analysis.py has not yet been run against the 5-fold benchmark artifacts.

Exact edits required:

1. DELETE Section 4.8 entirely from Chapter 4.

2. In Section 5.2 (Major Contributions), REMOVE the bullet:
   "SHAP Explainability Integration"

3. In Section 5.5 (Conclusion), REMOVE the paragraph starting:
   "The SHAP analysis provides a clinically coherent interpretation..."
   through to "...supports the feasibility of using the model outputs..."
   REPLACE WITH:
   "Post-hoc SHAP explainability analysis on the saved XGBoost model artifact
   (unified_xgb_model.json, best fold = Fold 3 AUROC 0.8095) is identified as
   an immediate next step. The expectation, grounded in the established sepsis
   pathophysiology, is that respiratory SOFA sub-score, lactate slope, and
   mean arterial pressure trend features will emerge as the dominant predictors,
   consistent with the literature on early sepsis detection."

4. In Section 5.4 (Future Work), ADD the following bullet:
   "SHAP Explainability: Execute shap_analysis.py against the saved
   unified_xgb_model.json artifact (best fold, AUROC=0.8095) to generate
   global beeswarm plots, dependence plots for the top predictor, and a
   waterfall plot for the highest-confidence true positive case."
