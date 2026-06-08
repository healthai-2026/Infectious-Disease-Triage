import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define directories
DATA_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\mimic-iv-clinical-database-demo-2.2"
PROCESSED_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Antibiotics regex patterns (to match prescriptions)
ABX_PATTERNS = [
    r'cillin', r'cef', r'cep', r'penem', r'floxacin', r'mycin', r'cycline',
    r'monam', r'oxacin', r'sulfamethoxazole', r'trimethoprim', r'metronidazole',
    r'linezolid', r'daptomycin', r'gentamicin', r'tobramycin', r'amikacin',
    r'aztreonam', r'clindamycin', r'levofloxacin', r'ciprofloxacin', r'moxifloxacin',
    r'piperacillin', r'tazobactam', r'meropenem', r'imipenem', r'ertapenem',
    r'ampicillin', r'nafcillin', r'oxacillin', r'cefazolin', r'ceftriaxone',
    r'cefepime', r'ceftazidime', r'azithromycin', r'clarithromycin', r'erythromycin',
    r'doxycycline', r'minocycline', r'tigecycline', r'colistin', r'nitrofurantoin'
]
NON_ABX = ['cepacol', 'cepastat', 'racepinephrine', 'epinephrine']

def load_csv_gz(subdir, filename):
    filepath = os.path.join(DATA_DIR, subdir, filename)
    print(f"Loading {filename}...")
    return pd.read_csv(filepath)

def identify_suspected_infections(df_micro, df_rx):
    """
    Identifies suspected infections based on:
    - Microbiology culture order (charttime)
    - Antibiotic prescription within [charttime - 24h, charttime + 72h]
    Returns suspected infection (SI) times per hadm_id.
    """
    print("Identifying suspected infections at hadm_id level...")
    # Parse times
    df_micro = df_micro.copy()
    df_micro['charttime'] = pd.to_datetime(df_micro['charttime'].fillna(df_micro['chartdate']))
    
    df_rx = df_rx.copy()
    df_rx['starttime'] = pd.to_datetime(df_rx['starttime'])
    
    # Filter for antibiotics
    pattern = '|'.join(ABX_PATTERNS)
    # Filter drugs matching regex
    df_rx_abx = df_rx[df_rx['drug'].str.contains(pattern, case=False, na=False)].copy()
    # Filter out false positives
    for non_ab in NON_ABX:
        df_rx_abx = df_rx_abx[~df_rx_abx['drug'].str.contains(non_ab, case=False, na=False)]
        
    si_records = []
    
    # Group by hadm_id to optimize search
    micro_grouped = df_micro.groupby('hadm_id')
    rx_grouped = df_rx_abx.groupby('hadm_id')
    
    for hadm_id, micro_group in micro_grouped:
        if pd.isna(hadm_id) or hadm_id not in rx_grouped.groups:
            continue
        rx_group = rx_grouped.get_group(hadm_id)
        
        for _, micro_row in micro_group.iterrows():
            t_culture = micro_row['charttime']
            # Find matching antibiotics
            valid_abx = rx_group[
                (rx_group['starttime'] >= t_culture - pd.Timedelta(hours=24)) &
                (rx_group['starttime'] <= t_culture + pd.Timedelta(hours=72))
            ]
            for _, rx_row in valid_abx.iterrows():
                t_abx = rx_row['starttime']
                t_si = min(t_culture, t_abx)
                si_records.append({
                    'subject_id': micro_row['subject_id'],
                    'hadm_id': hadm_id,
                    't_culture': t_culture,
                    't_abx': t_abx,
                    't_si': t_si
                })
                
    df_si = pd.DataFrame(si_records)
    if not df_si.empty:
        # Keep the earliest suspected infection per hadm_id
        df_si = df_si.sort_values(by='t_si').groupby('hadm_id').first().reset_index()
    print(f"Found {len(df_si)} admissions with suspected infection.")
    return df_si

def calculate_renal_sofa(df_creat, df_uop, stay_id, grid_times):
    """
    Computes renal SOFA score:
    - Creatinine (max value up to t)
    - Urine output (cumulative sum in the last 24 hours)
    """
    creat_scores = np.zeros(len(grid_times))
    if not df_creat.empty:
        df_creat = df_creat.sort_values('charttime')
        creat_idx = 0
        current_creat = 0.8
        for i, t in enumerate(grid_times):
            while creat_idx < len(df_creat) and df_creat.iloc[creat_idx]['charttime'] <= t:
                current_creat = df_creat.iloc[creat_idx]['valuenum']
                creat_idx += 1
            if current_creat >= 5.0:
                creat_scores[i] = 4
            elif current_creat >= 3.5:
                creat_scores[i] = 3
            elif current_creat >= 2.0:
                creat_scores[i] = 2
            elif current_creat >= 1.2:
                creat_scores[i] = 1
            else:
                creat_scores[i] = 0

    uop_scores = np.zeros(len(grid_times))
    if not df_uop.empty:
        df_uop = df_uop.sort_values('charttime')
        uop_times = df_uop['charttime'].values
        uop_vals = df_uop['value'].values
        
        for i, t in enumerate(grid_times):
            t_start = t - pd.Timedelta(hours=24)
            valid_mask = (uop_times > t_start) & (uop_times <= t)
            total_uop = uop_vals[valid_mask].sum()
            
            hours_since_start = (t - grid_times[0]).total_seconds() / 3600.0
            if hours_since_start < 24.0:
                if hours_since_start >= 4.0:
                    total_uop = total_uop * (24.0 / hours_since_start)
                else:
                    total_uop = 1000.0
            
            if total_uop < 200:
                uop_scores[i] = 4
            elif total_uop < 500:
                uop_scores[i] = 3
            else:
                uop_scores[i] = 0
                
    return np.maximum(creat_scores, uop_scores)

def calculate_cardio_sofa(df_map, df_vaso, grid_times):
    """
    Computes cardiovascular SOFA score:
    - MAP (mean arterial pressure)
    - Vasopressor infusion rate
    """
    cardio_scores = np.zeros(len(grid_times))
    
    vaso_active = []
    if not df_vaso.empty:
        for _, row in df_vaso.iterrows():
            vaso_active.append({
                'start': row['starttime'],
                'end': row['endtime'],
                'itemid': row['itemid'],
                'rate': row['rate'] if not pd.isna(row['rate']) else 0.0
            })
            
    map_idx = 0
    current_map = 80.0
    if not df_map.empty:
        df_map = df_map.sort_values('charttime')
        
    for i, t in enumerate(grid_times):
        if not df_map.empty:
            while map_idx < len(df_map) and df_map.iloc[map_idx]['charttime'] <= t:
                val = df_map.iloc[map_idx]['valuenum']
                if not pd.isna(val) and 30 < val < 200:
                    current_map = val
                map_idx += 1
                
        active_vasos = [v for v in vaso_active if v['start'] <= t <= v['end']]
        
        if len(active_vasos) > 0:
            max_score = 2
            for vaso in active_vasos:
                itemid = vaso['itemid']
                rate = vaso['rate']
                if itemid == 221662:
                    if rate > 15:
                        max_score = max(max_score, 4)
                    elif rate > 5:
                        max_score = max(max_score, 3)
                    else:
                        max_score = max(max_score, 2)
                elif itemid in [221906, 221289, 229617]:
                    if rate > 0.1:
                        max_score = max(max_score, 4)
                    else:
                        max_score = max(max_score, 3)
                elif itemid == 222315:
                    max_score = max(max_score, 4)
                elif itemid == 221653:
                    max_score = max(max_score, 2)
            cardio_scores[i] = max_score
        else:
            if current_map < 70.0:
                cardio_scores[i] = 1
            else:
                cardio_scores[i] = 0
                
    return cardio_scores

def calculate_cns_sofa(df_gcs, grid_times):
    """
    Computes neurological (CNS) SOFA score using GCS
    """
    cns_scores = np.zeros(len(grid_times))
    if df_gcs.empty:
        return cns_scores
        
    df_gcs = df_gcs.sort_values('charttime')
    
    df_gcs_pivot = df_gcs.pivot_table(index='charttime', columns='itemid', values='valuenum', aggfunc='last').reset_index()
    
    df_gcs_pivot = df_gcs_pivot.ffill().fillna({220739: 4, 223900: 5, 223901: 6})
    df_gcs_pivot['gcs'] = df_gcs_pivot[220739] + df_gcs_pivot[223900] + df_gcs_pivot[223901]
    
    gcs_times = df_gcs_pivot['charttime'].values
    gcs_vals = df_gcs_pivot['gcs'].values
    
    gcs_idx = 0
    current_gcs = 15.0
    for i, t in enumerate(grid_times):
        while gcs_idx < len(df_gcs_pivot) and gcs_times[gcs_idx] <= t:
            current_gcs = gcs_vals[gcs_idx]
            gcs_idx += 1
            
        if current_gcs < 6:
            cns_scores[i] = 4
        elif current_gcs < 10:
            cns_scores[i] = 3
        elif current_gcs < 13:
            cns_scores[i] = 2
        elif current_gcs < 15:
            cns_scores[i] = 1
        else:
            cns_scores[i] = 0
            
    return cns_scores

def calculate_resp_sofa(df_resp, grid_times):
    """
    Computes respiratory SOFA score:
    - PaO2/FiO2 ratio
    - Or SpO2/FiO2 ratio if PaO2 is missing
    """
    resp_scores = np.zeros(len(grid_times))
    if df_resp.empty:
        return resp_scores
        
    df_resp = df_resp.sort_values('charttime')
    df_pivot = df_resp.pivot_table(index='charttime', columns='itemid', values='valuenum', aggfunc='last').reset_index()
    
    if 223835 in df_pivot.columns:
        df_pivot[223835] = df_pivot[223835].apply(lambda x: x / 100.0 if x > 1.0 else x)
        df_pivot[223835] = df_pivot[223835].clip(0.21, 1.0)
    else:
        df_pivot[223835] = np.nan
        
    df_pivot = df_pivot.ffill().fillna({223835: 0.21})
    
    if 220277 not in df_pivot.columns:
        df_pivot[220277] = np.nan
    df_pivot[220277] = df_pivot[220277].fillna(98.0)
    
    if 50821 not in df_pivot.columns:
        df_pivot[50821] = np.nan
        
    resp_times = df_pivot['charttime'].values
    
    resp_idx = 0
    current_pao2 = np.nan
    current_spo2 = 98.0
    current_fio2 = 0.21
    
    for i, t in enumerate(grid_times):
        while resp_idx < len(df_pivot) and resp_times[resp_idx] <= t:
            row = df_pivot.iloc[resp_idx]
            current_pao2 = row[50821] if 50821 in row else np.nan
            current_spo2 = row[220277] if 220277 in row else 98.0
            current_fio2 = row[223835] if 223835 in row else 0.21
            resp_idx += 1
            
        pf_ratio = current_pao2 / current_fio2 if not pd.isna(current_pao2) else np.nan
        sf_ratio = current_spo2 / current_fio2
        
        if not pd.isna(pf_ratio):
            if pf_ratio < 100:
                resp_scores[i] = 4
            elif pf_ratio < 200:
                resp_scores[i] = 3
            elif pf_ratio < 300:
                resp_scores[i] = 2
            elif pf_ratio < 400:
                resp_scores[i] = 1
            else:
                resp_scores[i] = 0
        else:
            if sf_ratio < 150:
                resp_scores[i] = 4
            elif sf_ratio < 235:
                resp_scores[i] = 3
            elif sf_ratio < 315:
                resp_scores[i] = 2
            elif sf_ratio < 400:
                resp_scores[i] = 1
            else:
                resp_scores[i] = 0
                
    return resp_scores

def calculate_coag_sofa(df_plat, grid_times):
    """
    Computes coagulation SOFA score using platelets
    """
    coag_scores = np.zeros(len(grid_times))
    if df_plat.empty:
        return coag_scores
        
    df_plat = df_plat.sort_values('charttime')
    plat_times = df_plat['charttime'].values
    plat_vals = df_plat['valuenum'].values
    
    plat_idx = 0
    current_plat = 200.0
    for i, t in enumerate(grid_times):
        while plat_idx < len(df_plat) and plat_times[plat_idx] <= t:
            current_plat = plat_vals[plat_idx]
            plat_idx += 1
            
        if current_plat < 20:
            coag_scores[i] = 4
        elif current_plat < 50:
            coag_scores[i] = 3
        elif current_plat < 100:
            coag_scores[i] = 2
        elif current_plat < 150:
            coag_scores[i] = 1
        else:
            coag_scores[i] = 0
            
    return coag_scores

def calculate_liver_sofa(df_bili, grid_times):
    """
    Computes liver SOFA score using bilirubin
    """
    liver_scores = np.zeros(len(grid_times))
    if df_bili.empty:
        return liver_scores
        
    df_bili = df_bili.sort_values('charttime')
    bili_times = df_bili['charttime'].values
    bili_vals = df_bili['valuenum'].values
    
    bili_idx = 0
    current_bili = 0.5
    for i, t in enumerate(grid_times):
        while bili_idx < len(df_bili) and bili_times[bili_idx] <= t:
            current_bili = bili_vals[bili_idx]
            bili_idx += 1
            
        if current_bili >= 12.0:
            liver_scores[i] = 4
        elif current_bili >= 6.0:
            liver_scores[i] = 3
        elif current_bili >= 2.0:
            liver_scores[i] = 2
        elif current_bili >= 1.2:
            liver_scores[i] = 1
        else:
            liver_scores[i] = 0
            
    return liver_scores

def compute_slope(y):
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n)
    denom = n * np.sum(x**2) - np.sum(x)**2
    if denom == 0:
        return 0.0
    num = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    return num / denom

def build_features_and_labels(df_grid, stay_si_dict, lookback_hours=6):
    """
    Constructs the feature matrix and target labels for the stays.
    Lookback window: lookback_hours (default 6)
    Prediction window: next 6 hours
    """
    print(f"Building feature matrix and target labels (lookback={lookback_hours}h)...")
    feature_rows = []
    
    grouped_stays = df_grid.groupby('stay_id')
    sepsis_onsets = {}
    
    for stay_id, df_stay in grouped_stays:
        t_si = stay_si_dict.get(stay_id, None)
        if t_si is None:
            continue
            
        t_si = pd.to_datetime(t_si)
        win_start = t_si - pd.Timedelta(hours=48)
        win_end = t_si + pd.Timedelta(hours=24)
        
        df_win = df_stay[(df_stay['time'] >= win_start) & (df_stay['time'] <= win_end)].sort_values('time')
        if df_win.empty:
            continue
            
        sofa_vals = df_win['sofa_total'].values
        times = df_win['time'].values
        
        onset_time = None
        for idx in range(len(df_win)):
            current_sofa = sofa_vals[idx]
            current_time = times[idx]
            baseline_sofa = np.min(sofa_vals[:idx+1])
            
            if current_sofa - baseline_sofa >= 2:
                onset_time = current_time
                break
                
        if onset_time is not None:
            sepsis_onsets[stay_id] = onset_time
            
    print(f"Total stays with Sepsis-3 onset: {len(sepsis_onsets)} out of {len(grouped_stays)}")
    
    vitals_cols = ['heart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp', 'temp']
    labs_cols = ['glucose', 'potassium', 'sodium', 'creatinine', 'chloride', 'bun', 
                 'hematocrit', 'bicarbonate', 'platelets', 'hemoglobin', 'wbc', 'bilirubin', 'lactate']
    
    for stay_id, df_stay in grouped_stays:
        df_stay = df_stay.sort_values('time').reset_index(drop=True)
        subject_id = df_stay.iloc[0]['subject_id']
        age = df_stay.iloc[0]['age']
        gender = df_stay.iloc[0]['gender']
        onset_time = sepsis_onsets.get(stay_id, None)
        
        # Start building features after lookback_hours
        for idx in range(lookback_hours - 1, len(df_stay)):
            current_time = df_stay.iloc[idx]['time']
            
            # If sepsis onset happened at or before current_time, we exclude it
            if onset_time is not None and current_time >= onset_time:
                continue
                
            df_lookback = df_stay.iloc[idx - (lookback_hours - 1):idx + 1]
            current_row = df_stay.iloc[idx]
            
            row_features = {
                'stay_id': stay_id,
                'subject_id': subject_id,
                'time': current_time,
                'age': age,
                'gender': 1 if gender == 'M' else 0,
                'hours_since_admit': idx,
                'current_sofa': current_row['sofa_total'],
                'sofa_resp': current_row['sofa_resp'],
                'sofa_coag': current_row['sofa_coag'],
                'sofa_liver': current_row['sofa_liver'],
                'sofa_cardio': current_row['sofa_cardio'],
                'sofa_cns': current_row['sofa_cns'],
                'sofa_renal': current_row['sofa_renal']
            }
            
            # Vitals statistical summary
            for col in vitals_cols:
                vals = df_lookback[col].values
                row_features[f'{col}_mean'] = np.mean(vals)
                row_features[f'{col}_std'] = np.std(vals)
                row_features[f'{col}_min'] = np.min(vals)
                row_features[f'{col}_max'] = np.max(vals)
                row_features[f'{col}_last'] = vals[-1]
                row_features[f'{col}_slope'] = compute_slope(vals)
                
            # Labs (most recent value)
            for col in labs_cols:
                vals = df_lookback[col].values
                row_features[f'{col}_last'] = vals[-1]
                
            # Target Label: Will they develop sepsis in the next 6 hours?
            is_sepsis_stay = (onset_time is not None)
            if is_sepsis_stay:
                hours_to_onset = (onset_time - current_time).total_seconds() / 3600.0
                if 0.0 < hours_to_onset <= 6.0:
                    label = 1
                else:
                    label = 0
            else:
                label = 0
                
            row_features['label'] = label
            feature_rows.append(row_features)
            
    df_features = pd.DataFrame(feature_rows)
    print(f"Feature matrix built: {df_features.shape}")
    print(f"Sepsis positive samples (label=1): {df_features['label'].sum()} ({df_features['label'].mean()*100:.2f}%)")
    return df_features, sepsis_onsets

def main():
    print("Starting Preprocessing Pipeline...")
    
    # 1. Load data
    df_patients = load_csv_gz("hosp", "patients.csv.gz")
    df_admissions = load_csv_gz("hosp", "admissions.csv.gz")
    df_icustays = load_csv_gz("icu", "icustays.csv.gz")
    df_micro = load_csv_gz("hosp", "microbiologyevents.csv.gz")
    df_rx = load_csv_gz("hosp", "prescriptions.csv.gz")
    df_chartevents = load_csv_gz("icu", "chartevents.csv.gz")
    df_labevents = load_csv_gz("hosp", "labevents.csv.gz")
    df_vaso = load_csv_gz("icu", "inputevents.csv.gz")
    df_uop = load_csv_gz("icu", "outputevents.csv.gz")
    
    # Filter vasopressors
    vaso_itemids = [221906, 221289, 229617, 221662, 221653, 222315]
    df_vaso = df_vaso[df_vaso['itemid'].isin(vaso_itemids)].copy()
    df_vaso['starttime'] = pd.to_datetime(df_vaso['starttime'])
    df_vaso['endtime'] = pd.to_datetime(df_vaso['endtime'])
    
    # Filter urine outputs
    uop_itemids = [226559, 226566, 226627, 226631]
    df_uop = df_uop[df_uop['itemid'].isin(uop_itemids)].copy()
    df_uop['charttime'] = pd.to_datetime(df_uop['charttime'])
    
    # 2. Suspected Infection
    df_si = identify_suspected_infections(df_micro, df_rx)
    
    # Match with icustays at admission level
    df_merged = pd.merge(df_icustays, df_si, on=['subject_id', 'hadm_id'], how='inner')
    df_merged['intime'] = pd.to_datetime(df_merged['intime'])
    df_merged['outtime'] = pd.to_datetime(df_merged['outtime'])
    df_merged['t_si'] = pd.to_datetime(df_merged['t_si'])
    
    # Filter for valid stays where t_si falls in [intime - 24h, outtime]
    df_valid_si = df_merged[
        (df_merged['t_si'] >= df_merged['intime'] - pd.Timedelta(hours=24)) &
        (df_merged['t_si'] <= df_merged['outtime'])
    ]
    stay_si_dict = dict(zip(df_valid_si['stay_id'], df_valid_si['t_si']))
    print(f"Stays with valid suspected infection window: {len(stay_si_dict)}")
    
    # 3. Compile Vitals & Labs dataframes
    df_chartevents['charttime'] = pd.to_datetime(df_chartevents['charttime'])
    
    vitals_mapping = {
        220045: 'heart_rate',
        220210: 'resp_rate',
        220277: 'spo2',
        220179: 'sbp_ni', 220050: 'sbp_art',
        220180: 'dbp_ni', 220051: 'dbp_art',
        220181: 'mbp_ni', 220052: 'mbp_art',
        223761: 'temp_f', 223762: 'temp_c'
    }
    
    gcs_itemids = [220739, 223900, 223901]
    resp_itemids = [223835, 220277]
    
    # Filter chartevents
    all_icu_items = list(vitals_mapping.keys()) + gcs_itemids + resp_itemids
    df_icu_data = df_chartevents[df_chartevents['itemid'].isin(all_icu_items)].copy()
    
    df_labevents['charttime'] = pd.to_datetime(df_labevents['charttime'])
    
    labs_mapping = {
        50931: 'glucose', 52027: 'glucose', 50809: 'glucose',
        50971: 'potassium', 52452: 'potassium', 50822: 'potassium',
        50983: 'sodium', 52455: 'sodium', 50824: 'sodium',
        50912: 'creatinine', 52024: 'creatinine',
        50902: 'chloride',
        51006: 'bun',
        51221: 'hematocrit',
        50882: 'bicarbonate',
        51265: 'platelets', 51704: 'platelets',
        51222: 'hemoglobin',
        51301: 'wbc', 51300: 'wbc', 51755: 'wbc', 51756: 'wbc',
        50885: 'bilirubin',
        50813: 'lactate'
    }
    
    df_hosp_data = df_labevents[df_labevents['itemid'].isin(labs_mapping.keys())].copy()
    
    # 4. Process Stays Hourly
    grid_rows = []
    
    for _, stay in df_icustays.iterrows():
        stay_id = stay['stay_id']
        subject_id = stay['subject_id']
        hadm_id = stay['hadm_id']
        
        pat_row = df_patients[df_patients['subject_id'] == subject_id]
        if pat_row.empty:
            continue
        age = pat_row.iloc[0]['anchor_age']
        gender = pat_row.iloc[0]['gender']
        
        intime = pd.to_datetime(stay['intime'])
        outtime = pd.to_datetime(stay['outtime'])
        
        start_hour = intime.replace(minute=0, second=0, microsecond=0)
        end_hour = outtime.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
        grid_times = pd.date_range(start=start_hour, end=end_hour, freq='h')
        
        if len(grid_times) < 6: # We need at least 6 hours of data for lookback=6
            continue
            
        df_stay_icu = df_icu_data[df_icu_data['stay_id'] == stay_id].copy()
        df_stay_hosp = df_hosp_data[
            (df_hosp_data['subject_id'] == subject_id) &
            (df_hosp_data['charttime'] >= intime - pd.Timedelta(hours=24)) &
            (df_hosp_data['charttime'] <= outtime)
        ].copy()
        
        # Vitals
        vitals_dict = {col: np.full(len(grid_times), np.nan) for col in vitals_mapping.values()}
        vitals_dict['sbp'] = np.full(len(grid_times), np.nan)
        vitals_dict['dbp'] = np.full(len(grid_times), np.nan)
        vitals_dict['mbp'] = np.full(len(grid_times), np.nan)
        vitals_dict['temp'] = np.full(len(grid_times), np.nan)
        
        if not df_stay_icu.empty:
            df_vits_only = df_stay_icu[df_stay_icu['itemid'].isin(vitals_mapping.keys())].sort_values('charttime')
            for col_name in vitals_mapping.values():
                ids = [k for k, v in vitals_mapping.items() if v == col_name]
                df_var = df_vits_only[df_vits_only['itemid'].isin(ids)]
                if not df_var.empty:
                    var_times = df_var['charttime'].values
                    var_vals = df_var['valuenum'].values
                    var_idx = 0
                    current_val = np.nan
                    for i, t in enumerate(grid_times):
                        while var_idx < len(df_var) and var_times[var_idx] <= t:
                            current_val = var_vals[var_idx]
                            var_idx += 1
                        vitals_dict[col_name][i] = current_val
                        
            for i in range(len(grid_times)):
                sbp_ni = vitals_dict['sbp_ni'][i]
                sbp_art = vitals_dict['sbp_art'][i]
                vitals_dict['sbp'][i] = sbp_art if not pd.isna(sbp_art) else sbp_ni
                
                dbp_ni = vitals_dict['dbp_ni'][i]
                dbp_art = vitals_dict['dbp_art'][i]
                vitals_dict['dbp'][i] = dbp_art if not pd.isna(dbp_art) else dbp_ni
                
                mbp_ni = vitals_dict['mbp_ni'][i]
                mbp_art = vitals_dict['mbp_art'][i]
                vitals_dict['mbp'][i] = mbp_art if not pd.isna(mbp_art) else mbp_ni
                
                temp_f = vitals_dict['temp_f'][i]
                temp_c = vitals_dict['temp_c'][i]
                if not pd.isna(temp_c):
                    vitals_dict['temp'][i] = temp_c * 1.8 + 32.0
                elif not pd.isna(temp_f):
                    vitals_dict['temp'][i] = temp_f
                    
        vitals_defaults = {'heart_rate': 80.0, 'resp_rate': 15.0, 'spo2': 98.0, 'sbp': 120.0, 'dbp': 80.0, 'mbp': 80.0, 'temp': 98.6}
        for col in ['heart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp', 'temp']:
            s = pd.Series(vitals_dict[col]).ffill().bfill().fillna(vitals_defaults[col])
            vitals_dict[col] = s.values
            
        # Labs
        labs_dict = {col: np.full(len(grid_times), np.nan) for col in set(labs_mapping.values())}
        if not df_stay_hosp.empty:
            df_labs_only = df_stay_hosp.sort_values('charttime')
            for col_name in set(labs_mapping.values()):
                ids = [k for k, v in labs_mapping.items() if v == col_name]
                df_var = df_labs_only[df_labs_only['itemid'].isin(ids)]
                if not df_var.empty:
                    var_times = df_var['charttime'].values
                    var_vals = df_var['valuenum'].values
                    var_idx = 0
                    current_val = np.nan
                    for i, t in enumerate(grid_times):
                        while var_idx < len(df_var) and var_times[var_idx] <= t:
                            current_val = var_vals[var_idx]
                            var_idx += 1
                        labs_dict[col_name][i] = current_val
                        
        labs_defaults = {
            'glucose': 100.0, 'potassium': 4.0, 'sodium': 140.0, 'creatinine': 0.8,
            'chloride': 100.0, 'bun': 15.0, 'hematocrit': 40.0, 'bicarbonate': 24.0,
            'platelets': 200.0, 'hemoglobin': 13.0, 'wbc': 7.0, 'bilirubin': 0.5, 'lactate': 1.0
        }
        for col in labs_defaults.keys():
            s = pd.Series(labs_dict[col]).ffill().bfill().fillna(labs_defaults[col])
            labs_dict[col] = s.values

        # SOFA components
        df_pao2 = df_stay_hosp[df_stay_hosp['itemid'] == 50821].copy()
        df_resp_icu = df_stay_icu[df_stay_icu['itemid'].isin([220277, 223835])].copy()
        df_resp = pd.concat([df_pao2[['charttime', 'itemid', 'valuenum']], df_resp_icu[['charttime', 'itemid', 'valuenum']]])
        sofa_resp = calculate_resp_sofa(df_resp, grid_times)
        
        df_plat = df_stay_hosp[df_stay_hosp['itemid'].isin([51265, 51704])].copy()
        sofa_coag = calculate_coag_sofa(df_plat, grid_times)
        
        df_bili = df_stay_hosp[df_stay_hosp['itemid'] == 50885].copy()
        sofa_liver = calculate_liver_sofa(df_bili, grid_times)
        
        df_map = df_stay_icu[df_stay_icu['itemid'].isin([220052, 220181])].copy()
        df_stay_vaso = df_vaso[df_vaso['stay_id'] == stay_id].copy()
        sofa_cardio = calculate_cardio_sofa(df_map, df_stay_vaso, grid_times)
        
        df_gcs = df_stay_icu[df_stay_icu['itemid'].isin(gcs_itemids)].copy()
        sofa_cns = calculate_cns_sofa(df_gcs, grid_times)
        
        df_creat = df_stay_hosp[df_stay_hosp['itemid'].isin([50912, 52024])].copy()
        df_stay_uop = df_uop[df_uop['stay_id'] == stay_id].copy()
        sofa_renal = calculate_renal_sofa(df_creat, df_stay_uop, stay_id, grid_times)
        
        sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal
        
        for i, t in enumerate(grid_times):
            row = {
                'stay_id': stay_id,
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'time': t,
                'age': age,
                'gender': gender,
                'sofa_resp': sofa_resp[i],
                'sofa_coag': sofa_coag[i],
                'sofa_liver': sofa_liver[i],
                'sofa_cardio': sofa_cardio[i],
                'sofa_cns': sofa_cns[i],
                'sofa_renal': sofa_renal[i],
                'sofa_total': sofa_total[i]
            }
            for col in vitals_defaults.keys():
                row[col] = vitals_dict[col][i]
            for col in labs_defaults.keys():
                row[col] = labs_dict[col][i]
                
            grid_rows.append(row)
            
    df_grid = pd.DataFrame(grid_rows)
    print(f"Hourly grid built: {df_grid.shape}")
    
    # 5. Extract Feature Matrix and Labels (lookback=6)
    df_features, sepsis_onsets = build_features_and_labels(df_grid, stay_si_dict, lookback_hours=6)
    
    # Save datasets
    df_features.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)
    
    # Save metadata/stats
    stats = {
        'total_patients': int(df_patients['subject_id'].nunique()),
        'total_stays': int(df_icustays['stay_id'].nunique()),
        'stays_in_grid': int(df_grid['stay_id'].nunique()),
        'stays_with_sepsis': len(sepsis_onsets),
        'total_rows_features': len(df_features),
        'positive_rows': int(df_features['label'].sum()),
        'negative_rows': int(len(df_features) - df_features['label'].sum())
    }
    
    with open(os.path.join(PROCESSED_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Preprocessing completed successfully!")
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    main()
