import pandas as pd
import numpy as np
import os
import time

# Configurations
MIMIC_PATH = r"D:\Internship2026\Infectious-Disease-Triage\Data\mimic-iv-3.1"
OUTPUT_PATH = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed\sepsis_features.csv"

def main():
    start_time = time.time()
    print("Step 1: Identifying Sepsis admissions from ICD codes...")
    diagnoses_path = os.path.join(MIMIC_PATH, "hosp/diagnoses_icd.csv.gz")
    diagnoses = pd.read_csv(diagnoses_path)
    
    # Strip and convert to string for robust matching
    diagnoses['icd_code'] = diagnoses['icd_code'].astype(str).str.strip()
    
    # Define Sepsis ICD codes:
    # ICD-9: starts with 038, or is 99591, 99592, 78552
    # ICD-10: starts with A40, A41, or starts with R652
    is_sepsis_9 = (diagnoses['icd_version'] == 9) & (
        diagnoses['icd_code'].str.startswith('038') | 
        diagnoses['icd_code'].isin(['99591', '99592', '78552'])
    )
    is_sepsis_10 = (diagnoses['icd_version'] == 10) & (
        diagnoses['icd_code'].str.startswith('A40') | 
        diagnoses['icd_code'].str.startswith('A41') | 
        diagnoses['icd_code'].str.startswith('R652')
    )
    
    sepsis_hadms = set(diagnoses[is_sepsis_9 | is_sepsis_10]['hadm_id'].unique())
    print(f"Found {len(sepsis_hadms)} sepsis admissions in diagnoses_icd.csv.")

    print("\nStep 2: Loading ICU stays and merging demographics...")
    icustays = pd.read_csv(os.path.join(MIMIC_PATH, "icu/icustays.csv.gz"))
    print(f"Total ICU stays loaded: {len(icustays)}")
    
    # Parse in time
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    # Get patients info
    patients = pd.read_csv(os.path.join(MIMIC_PATH, "hosp/patients.csv.gz"), 
                           usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    
    df_stays = icustays.merge(patients, on='subject_id', how='left')
    
    # Calculate exact age at time of ICU admission
    df_stays['age'] = df_stays['anchor_age'] + (df_stays['intime'].dt.year - df_stays['anchor_year'])
    df_stays.loc[df_stays['age'] < 0, 'age'] = df_stays.loc[df_stays['age'] < 0, 'anchor_age']
    
    # Load admissions for race and admission type
    admissions = pd.read_csv(os.path.join(MIMIC_PATH, "hosp/admissions.csv.gz"), 
                            usecols=['hadm_id', 'race', 'admission_type'])
    df_stays = df_stays.merge(admissions, on='hadm_id', how='left')
    
    # Label Sepsis
    df_stays['sepsis'] = df_stays['hadm_id'].isin(sepsis_hadms).astype(int)
    
    print(f"Sepsis rate across all ICU stays: {df_stays['sepsis'].mean()*100:.2f}% ({df_stays['sepsis'].sum()} stays)")

    # Prepare lookup dictionary for stay_id -> intime to filter chartevents in the first 24h
    # Converting to numpy datetime64 makes operations extremely fast
    stay_intime = df_stays.set_index('stay_id')['intime'].to_dict()
    stay_set = set(stay_intime.keys())
    
    print("\nStep 3: Extracting vitals and labs from chartevents.csv...")
    # Feature item ID mappings
    item_to_feature = {
        220045: 'heart_rate',
        220179: 'sys_bp',
        220050: 'sys_bp',
        220180: 'dias_bp',
        220051: 'dias_bp',
        220181: 'mean_bp',
        220052: 'mean_bp',
        220210: 'respiratory_rate',
        223761: 'temp_f', # Fahrenheit
        223762: 'temp_c', # Celsius
        220277: 'spo2',
        220621: 'glucose',
        225664: 'glucose',
        226537: 'glucose',
        228388: 'glucose',
        220645: 'sodium',
        226534: 'sodium',
        228389: 'sodium',
        228390: 'sodium',
        220615: 'creatinine',
        229761: 'creatinine',
        220546: 'wbc'
    }
    target_item_ids = set(item_to_feature.keys())
    
    # Cleaning/clipping ranges to filter extreme outliers/errors
    clean_ranges = {
        'heart_rate': (20.0, 250.0),
        'sys_bp': (40.0, 280.0),
        'dias_bp': (20.0, 180.0),
        'mean_bp': (20.0, 200.0),
        'respiratory_rate': (4.0, 80.0),
        'temp_c': (25.0, 45.0),
        'spo2': (40.0, 100.0),
        'glucose': (10.0, 1000.0),
        'sodium': (90.0, 180.0),
        'creatinine': (0.1, 20.0),
        'wbc': (0.1, 150.0)
    }

    chartevents_path = os.path.join(MIMIC_PATH, "icu/chartevents.csv.gz")
    chunk_size = 2000000  # 2M rows chunks
    collected_data = []
    
    chunk_idx = 0
    total_valid_rows = 0
    
    print("Reading chartevents.csv in chunks...")
    # Read only required columns to save memory and time
    for chunk in pd.read_csv(chartevents_path, chunksize=chunk_size, 
                             usecols=['stay_id', 'charttime', 'itemid', 'valuenum']):
        chunk_idx += 1
        
        # Filter for relevant itemids and stay_ids immediately to reduce size
        sub = chunk[chunk['itemid'].isin(target_item_ids) & chunk['stay_id'].isin(stay_set)].copy()
        if sub.empty:
            continue
            
        # Parse datetime
        sub['charttime'] = pd.to_datetime(sub['charttime'])
        
        # Look up intime
        sub['intime'] = sub['stay_id'].map(stay_intime)
        
        # Compute hours since ICU admission
        sub['hours'] = (sub['charttime'] - sub['intime']).dt.total_seconds() / 3600.0
        
        # Filter: first 24 hours of ICU stay
        sub = sub[(sub['hours'] >= 0.0) & (sub['hours'] <= 24.0)]
        if sub.empty:
            continue
            
        # Map itemid to feature name
        sub['feature'] = sub['itemid'].map(item_to_feature)
        
        # Drop rows with missing values
        sub = sub.dropna(subset=['valuenum'])
        
        # Temperature conversion Fahrenheit -> Celsius
        temp_f_mask = sub['feature'] == 'temp_f'
        if temp_f_mask.any():
            sub.loc[temp_f_mask, 'valuenum'] = (sub.loc[temp_f_mask, 'valuenum'] - 32.0) * 5.0 / 9.0
            sub.loc[temp_f_mask, 'feature'] = 'temp_c'
            
        # Filter outliers
        for feat, (val_min, val_max) in clean_ranges.items():
            feat_mask = sub['feature'] == feat
            if feat_mask.any():
                sub = sub[~feat_mask | ((sub['valuenum'] >= val_min) & (sub['valuenum'] <= val_max))]
                
        if sub.empty:
            continue
            
        collected_data.append(sub[['stay_id', 'feature', 'valuenum']])
        total_valid_rows += len(sub)
        
        if chunk_idx % 5 == 0:
            print(f"Processed {chunk_idx * chunk_size // 1000000}M rows of chartevents.csv... Retained {total_valid_rows} measurements.")
            
    print(f"Finished reading chartevents.csv. Total measurements retained: {total_valid_rows}")
    
    if not collected_data:
        print("Error: No data was collected!")
        return
        
    print("\nStep 4: Aggregating measurements...")
    all_events = pd.concat(collected_data, ignore_index=True)
    
    # Aggregate to compute mean, min, max, count
    features_agg = all_events.groupby(['stay_id', 'feature'])['valuenum'].agg(['mean', 'min', 'max', 'count']).unstack()
    
    # Flatten multi-index columns
    features_agg.columns = [f"{feat}_{stat}" for stat, feat in features_agg.columns]
    features_agg = features_agg.reset_index()
    
    print("\nStep 5: Merging clinical features with stay demographics...")
    # Keep only relevant columns from stays
    stays_clean = df_stays[['stay_id', 'subject_id', 'hadm_id', 'gender', 'age', 'race', 'admission_type', 'sepsis']]
    
    # Left join to retain demographics even if some features are missing
    final_df = stays_clean.merge(features_agg, on='stay_id', how='left')
    
    # Let's count how many features are non-null
    feature_cols = [c for c in final_df.columns if c not in ['stay_id', 'subject_id', 'hadm_id', 'gender', 'age', 'race', 'admission_type', 'sepsis']]
    
    # We will filter out stays that have 0 clinical measurements in the first 24h
    non_null_counts = final_df[feature_cols].notnull().sum(axis=1)
    final_df_filtered = final_df[non_null_counts > 0].copy()
    
    print(f"Stays with at least one measurement: {len(final_df_filtered)} / {len(final_df)}")
    print(f"Sepsis rate in filtered stays: {final_df_filtered['sepsis'].mean()*100:.2f}% ({final_df_filtered['sepsis'].sum()} stays)")
    
    print(f"\nStep 6: Saving processed features to {OUTPUT_PATH}...")
    final_df_filtered.to_csv(OUTPUT_PATH, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Data extraction completed successfully in {duration/60.0:.2f} minutes!")

if __name__ == "__main__":
    main()
