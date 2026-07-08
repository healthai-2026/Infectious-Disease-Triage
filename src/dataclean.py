import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\mimic-iv-3.1"
PROCESSED_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading demographics and stays...")
patients = pd.read_csv(os.path.join(DATA_DIR, "hosp", "patients.csv.gz"))
admissions = pd.read_csv(os.path.join(DATA_DIR, "hosp", "admissions.csv.gz"))
icustays = pd.read_csv(os.path.join(DATA_DIR, "icu", "icustays.csv.gz"))

d_items = pd.read_csv(os.path.join(DATA_DIR, "icu", "d_items.csv.gz"))
d_labitems = pd.read_csv(os.path.join(DATA_DIR, "hosp", "d_labitems.csv.gz"))

patients = patients[['subject_id','gender','anchor_age']]

patients = patients.drop_duplicates()

patients = patients[
    (patients['anchor_age'] >= 0) &
    (patients['anchor_age'] <= 120)
]

patients['gender'] = patients['gender'].str.upper()

admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

admissions = admissions.drop_duplicates()

admissions = admissions[
    admissions['dischtime'] > admissions['admittime']
]

icustays['intime'] = pd.to_datetime(icustays['intime'])
icustays['outtime'] = pd.to_datetime(icustays['outtime'])

icustays = icustays.drop_duplicates()

icustays = icustays[
    icustays['outtime'] > icustays['intime']
]

d_items[d_items['label'].str.contains(
    'Heart Rate',
    case=False,
    na=False
)]

d_items[d_items['label'].str.contains(
    'Resp',
    case=False,
    na=False
)]

d_items[d_items['label'].str.contains(
    'Temperature',
    case=False,
    na=False
)]

d_items[d_items['label'].str.contains(
    'SpO2',
    case=False,
    na=False
)]

d_items[d_items['label'].str.contains(
    'Mean',
    case=False,
    na=False
)]

HR = [220045]
RR = [220210]
TEMP = [223762]
MAP = [220181]
SPO2 = [220277]

vital_ids = HR + RR + TEMP + MAP + SPO2

print("Loading and filtering chartevents.csv.gz...")
chartevents_chunks = []
chunk_idx = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "icu", "chartevents.csv.gz"), usecols=['stay_id', 'itemid', 'charttime', 'valuenum'], chunksize=1000000):
    chunk_idx += 1
    if chunk_idx % 10 == 0:
        print(f"  Processed {chunk_idx} million rows of chartevents...")
    chunk_filtered = chunk[chunk['itemid'].isin(vital_ids)]
    chartevents_chunks.append(chunk_filtered)
vitals = pd.concat(chartevents_chunks)

vitals['valuenum'] = pd.to_numeric(
    vitals['valuenum'],
    errors='coerce'
)

vitals = vitals.dropna(subset=['valuenum'])

vitals = vitals[
    ~(
        (vitals['itemid']==220045) &
        (
            (vitals['valuenum'] < 30) |
            (vitals['valuenum'] > 250)
        )
    )
]

vitals = vitals[
    ~(
        (vitals['itemid']==220210) &
        (
            (vitals['valuenum'] < 4) |
            (vitals['valuenum'] > 80)
        )
    )
]

vitals = vitals[
    ~(
        (vitals['itemid']==223762) &
        (
            (vitals['valuenum'] < 30) |
            (vitals['valuenum'] > 45)
        )
    )
]

vitals = vitals[
    ~(
        (vitals['itemid']==220277) &
        (
            (vitals['valuenum'] < 0) |
            (vitals['valuenum'] > 100)
        )
    )
]

wanted = [
    'Lactate',
    'Creatinine',
    'Platelet Count',
    'WBC',
    'Bilirubin',
    'Bicarbonate'
]

for lab in wanted:
    print(
        d_labitems[
            d_labitems['label'].str.contains(
                lab,
                case=False,
                na=False
            )
        ][['itemid','label']]
    )

lab_ids = [
    50813,   # Lactate
    50912,   # Creatinine
    51265,   # Platelets
    51300,   # WBC
    50885,   # Bilirubin
    50882    # Bicarbonate
]

print("Loading and filtering labevents.csv.gz...")
labevents_chunks = []
chunk_idx = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "hosp", "labevents.csv.gz"), usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=1000000):
    chunk_idx += 1
    if chunk_idx % 10 == 0:
        print(f"  Processed {chunk_idx} million rows of labevents...")
    chunk_filtered = chunk[chunk['itemid'].isin(lab_ids)]
    labevents_chunks.append(chunk_filtered)
labs = pd.concat(labevents_chunks)

labs['valuenum'] = pd.to_numeric(
    labs['valuenum'],
    errors='coerce'
)

labs = labs.dropna(subset=['valuenum'])

labs = labs[
    (labs['valuenum'] >= 0)
]

urine_ids = [
    226559,
    226560,
    226561
]

print("Loading and filtering outputevents.csv.gz...")
outputevents_chunks = []
chunk_idx = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "icu", "outputevents.csv.gz"), usecols=['stay_id', 'itemid', 'charttime', 'value'], chunksize=500000):
    chunk_idx += 1
    if chunk_idx % 10 == 0:
        print(f"  Processed {chunk_idx * 0.5:.1f} million rows of outputevents...")
    chunk_filtered = chunk[chunk['itemid'].isin(urine_ids)]
    outputevents_chunks.append(chunk_filtered)
urine = pd.concat(outputevents_chunks)

urine = urine[
    urine['value'] >= 0
]

vitals['charttime'] = pd.to_datetime(vitals['charttime'])

icu = icustays[['stay_id','intime']]

vitals = vitals.merge(
    icu,
    on='stay_id'
)

vitals = vitals[
    vitals['charttime']
    <=
    vitals['intime']
    + pd.Timedelta(hours=6)
]

mean_features = (
    vitals
    .groupby(['stay_id','itemid'])
    ['valuenum']
    .mean()
    .unstack()
)

max_features = (
    vitals
    .groupby(['stay_id','itemid'])
    ['valuenum']
    .max()
    .unstack()
)

min_features = (
    vitals
    .groupby(['stay_id','itemid'])
    ['valuenum']
    .min()
    .unstack()
)

std_features = (
    vitals
    .groupby(['stay_id','itemid'])
    ['valuenum']
    .std()
    .unstack()
)

# X is not defined yet, this is handled later.

for v in [
    'patients',
    'icustays',
    'vitals',
    'labs',
    'mean_features',
    'max_features',
    'min_features',
    'std_features',
    'X'
]:
    print(v, v in globals())

print(mean_features.shape)
print(max_features.shape)
print(min_features.shape)
print(std_features.shape)

mean_features.columns = [f"{c}_mean" for c in mean_features.columns]

max_features.columns = [f"{c}_max" for c in max_features.columns]

min_features.columns = [f"{c}_min" for c in min_features.columns]

std_features.columns = [f"{c}_std" for c in std_features.columns]

X = pd.concat(
    [
        mean_features,
        max_features,
        min_features,
        std_features
    ],
    axis=1
)

print(X.shape)
X.head()

X.isnull().sum().sort_values(ascending=False).head(20)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)


X.isnull().sum().sum()

X.to_csv(
    os.path.join(PROCESSED_DIR, "clean_feature_matrix.csv"),
    index=True
)

X.to_csv(
    os.path.join(PROCESSED_DIR, "sepsis_features_clean.csv"),
    index=False
)

print(X.shape)
X.head()

# Merge ICU identifiers
X = (
    X.reset_index()
    .merge(
        icustays[['stay_id', 'hadm_id', 'subject_id']],
        on='stay_id',
        how='left'
    )
    .merge(
        patients[['subject_id', 'anchor_age', 'gender']],
        on='subject_id',
        how='left'
    )
)

# Encode gender
X['gender'] = X['gender'].map({'M': 1, 'F': 0})

print(X.shape)
X.head()

plt.figure(figsize=(12,6))
(X.isnull().mean() * 100).plot(kind='bar')
plt.title("Missing Values Before Cleaning (Percentage)")
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "missing_values_before.png"))
plt.close()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

plt.figure(figsize=(12,6))
(X.isnull().mean() * 100).plot(kind='bar')
plt.title("Missing Values After Cleaning (Percentage)")
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "missing_values_after.png"))
plt.close()

plt.figure(figsize=(8,5))

X['220045_mean'].hist(bins=50)

plt.title("Heart Rate After Cleaning")
plt.savefig(os.path.join(PROCESSED_DIR, "heart_rate_distribution.png"))
plt.close()

print(X.columns.tolist())

for col in X.columns:
    print(col)

mean_features.columns = [f"{c}_mean" for c in mean_features.columns]
max_features.columns = [f"{c}_max" for c in max_features.columns]
min_features.columns = [f"{c}_min" for c in min_features.columns]
std_features.columns = [f"{c}_std" for c in std_features.columns]

[c for c in X.columns if '220045' in str(c)]

plt.figure(figsize=(8,5))

X['220045_mean'].hist(bins=50)

plt.title("Heart Rate Mean Distribution")
plt.savefig(os.path.join(PROCESSED_DIR, "heart_rate_mean_distribution.png"))
plt.close()

# Save final feature matrix with demographics
X.to_csv(
    os.path.join(PROCESSED_DIR, "sepsis_features_final.csv"),
    index=False
)
print("Saved sepsis_features_final.csv to processed directory.")