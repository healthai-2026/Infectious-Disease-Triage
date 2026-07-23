import json
import os
import random
import re
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\mimic-iv-3.1"
PROCESSED_DIR = r"C:\PS1\Infectious-Disease-Triage\Data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

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

print("Starting Preprocessing Pipeline...")

print("Loading patients.csv.gz...")
df_patients = pd.read_csv(os.path.join(DATA_DIR, "hosp", "patients.csv.gz"), usecols=['subject_id', 'gender', 'anchor_age'])
print("Loading admissions.csv.gz...")
df_admissions = pd.read_csv(os.path.join(DATA_DIR, "hosp", "admissions.csv.gz"), usecols=['subject_id', 'hadm_id'])
print("Loading icustays.csv.gz...")
df_icustays = pd.read_csv(os.path.join(DATA_DIR, "icu", "icustays.csv.gz"), usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
print("Loading microbiologyevents.csv.gz...")
df_micro = pd.read_csv(os.path.join(DATA_DIR, "hosp", "microbiologyevents.csv.gz"), usecols=['subject_id', 'hadm_id', 'charttime', 'chartdate'])

print("Loading and filtering prescriptions.csv.gz...")
pattern = '|'.join(ABX_PATTERNS)
rx_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "hosp", "prescriptions.csv.gz"), usecols=['subject_id', 'hadm_id', 'drug', 'starttime'], chunksize=500000):
    chunk_filtered = chunk[chunk['drug'].str.contains(pattern, case=False, na=False)].copy()
    for non_ab in NON_ABX:
        chunk_filtered = chunk_filtered[~chunk_filtered['drug'].str.contains(non_ab, case=False, na=False)]
    rx_chunks.append(chunk_filtered)
df_rx_abx = pd.concat(rx_chunks)
df_rx_abx['starttime'] = pd.to_datetime(df_rx_abx['starttime'])

print("Loading and filtering inputevents.csv.gz...")
vaso_itemids = [221906, 221289, 229617, 221662, 221653, 222315]
vaso_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "icu", "inputevents.csv.gz"), usecols=['stay_id', 'itemid', 'starttime', 'endtime', 'rate'], chunksize=500000):
    chunk_filtered = chunk[chunk['itemid'].isin(vaso_itemids)]
    vaso_chunks.append(chunk_filtered)
df_vaso = pd.concat(vaso_chunks)
df_vaso['starttime'] = pd.to_datetime(df_vaso['starttime'])
df_vaso['endtime'] = pd.to_datetime(df_vaso['endtime'])

print("Loading and filtering outputevents.csv.gz...")
uop_itemids = [226559, 226566, 226627, 226631]
uop_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "icu", "outputevents.csv.gz"), usecols=['stay_id', 'itemid', 'charttime', 'value'], chunksize=500000):
    chunk_filtered = chunk[chunk['itemid'].isin(uop_itemids)]
    uop_chunks.append(chunk_filtered)
df_uop = pd.concat(uop_chunks)
df_uop['charttime'] = pd.to_datetime(df_uop['charttime'])

print("Identifying suspected infections at hadm_id level...")
df_micro_temp = df_micro.copy()
df_micro_temp['charttime'] = pd.to_datetime(df_micro_temp['charttime'].fillna(df_micro_temp['chartdate']))

# Vectorized matching of microbiology cultures and antibiotic prescriptions
df_merged_micro_rx = pd.merge(
    df_micro_temp[['subject_id', 'hadm_id', 'charttime']], 
    df_rx_abx[['hadm_id', 'starttime']], 
    on='hadm_id'
)

# Filter for the infection suspected window: abx within [culture - 24h, culture + 72h]
df_valid_si_events = df_merged_micro_rx[
    (df_merged_micro_rx['starttime'] >= df_merged_micro_rx['charttime'] - pd.Timedelta(hours=24)) &
    (df_merged_micro_rx['starttime'] <= df_merged_micro_rx['charttime'] + pd.Timedelta(hours=72))
].copy()

if not df_valid_si_events.empty:
    df_valid_si_events['t_si'] = np.minimum(df_valid_si_events['charttime'], df_valid_si_events['starttime'])
    df_si = df_valid_si_events.rename(columns={'charttime': 't_culture', 'starttime': 't_abx'})
    df_si = df_si.sort_values(by='t_si').groupby('hadm_id').first().reset_index()
else:
    df_si = pd.DataFrame(columns=['subject_id', 'hadm_id', 't_culture', 't_abx', 't_si'])
print(f"Found {len(df_si)} admissions with suspected infection.")

df_merged = pd.merge(df_icustays, df_si, on=['subject_id', 'hadm_id'], how='inner')
df_merged['intime'] = pd.to_datetime(df_merged['intime'])
df_merged['outtime'] = pd.to_datetime(df_merged['outtime'])
df_merged['t_si'] = pd.to_datetime(df_merged['t_si'])

df_valid_si = df_merged[
    (df_merged['t_si'] >= df_merged['intime'] - pd.Timedelta(hours=24)) &
    (df_merged['t_si'] <= df_merged['outtime'])
]
stay_si_dict = dict(zip(df_valid_si['stay_id'], df_valid_si['t_si']))
print(f"Stays with valid suspected infection window: {len(stay_si_dict)}")

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

all_icu_items = list(vitals_mapping.keys()) + gcs_itemids + resp_itemids

print("Loading and filtering chartevents.csv.gz...")
chartevents_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "icu", "chartevents.csv.gz"), usecols=['stay_id', 'subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=1000000):
    chunk_filtered = chunk[chunk['itemid'].isin(all_icu_items) & chunk['valuenum'].notna()]
    chartevents_chunks.append(chunk_filtered)
df_icu_data = pd.concat(chartevents_chunks)
df_icu_data['charttime'] = pd.to_datetime(df_icu_data['charttime'])

# Apply simple physiologically plausible ranges for common vitals to reduce obvious data-entry errors.
vital_ranges = {
    220045: (20, 300),
    220210: (4, 60),
    220277: (60, 100),
    220179: (40, 300),
    220180: (20, 200),
    220181: (40, 300),
    223761: (80, 115),
    223762: (25, 45),
}
for itemid, (lo, hi) in vital_ranges.items():
    mask = (df_icu_data['itemid'] == itemid)
    df_icu_data = df_icu_data[~(mask & ((df_icu_data['valuenum'] < lo) | (df_icu_data['valuenum'] > hi)))]

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

print("Loading and filtering labevents.csv.gz...")
labevents_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "hosp", "labevents.csv.gz"), usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=1000000):
    chunk_filtered = chunk[chunk['itemid'].isin(labs_mapping.keys()) & chunk['valuenum'].notna()]
    labevents_chunks.append(chunk_filtered)
df_hosp_data = pd.concat(labevents_chunks)
df_hosp_data['charttime'] = pd.to_datetime(df_hosp_data['charttime'])

grid_rows = []

print("Preparing group maps for fast lookups...")
pat_dict = df_patients.set_index('subject_id')[['gender', 'anchor_age']].to_dict('index')

print("Preparing ICU data dict...")
icu_dict = {}
for row in df_icu_data.itertuples(index=False):
    sid = row.stay_id
    iid = row.itemid
    if sid not in icu_dict:
        icu_dict[sid] = {}
    if iid not in icu_dict[sid]:
        icu_dict[sid][iid] = {'times': [], 'vals': []}
    icu_dict[sid][iid]['times'].append(row.charttime)
    icu_dict[sid][iid]['vals'].append(row.valuenum)
    
for sid in icu_dict:
    for iid in icu_dict[sid]:
        icu_dict[sid][iid]['times'] = np.array(icu_dict[sid][iid]['times'])
        icu_dict[sid][iid]['vals'] = np.array(icu_dict[sid][iid]['vals'])

# Free up memory
del df_icu_data
import gc; gc.collect()

print("Preparing hospital labs dict...")
hosp_dict = {}
for row in df_hosp_data.itertuples(index=False):
    sub = row.subject_id
    iid = row.itemid
    if sub not in hosp_dict:
        hosp_dict[sub] = {}
    if iid not in hosp_dict[sub]:
        hosp_dict[sub][iid] = {'times': [], 'vals': []}
    hosp_dict[sub][iid]['times'].append(row.charttime)
    hosp_dict[sub][iid]['vals'].append(row.valuenum)

for sub in hosp_dict:
    for iid in hosp_dict[sub]:
        hosp_dict[sub][iid]['times'] = np.array(hosp_dict[sub][iid]['times'])
        hosp_dict[sub][iid]['vals'] = np.array(hosp_dict[sub][iid]['vals'])

# Free up memory
del df_hosp_data
gc.collect()

print("Preparing vasoactive dict...")
vaso_dict = {}
for stay_id, group in df_vaso.groupby('stay_id'):
    vaso_dict[stay_id] = [
        {
            'start': r.starttime,
            'end': r.endtime,
            'itemid': r.itemid,
            'rate': r.rate if not pd.isna(r.rate) else 0.0
        }
        for r in group.itertuples()
    ]

print("Preparing urine output dict...")
uop_dict = {}
for stay_id, group in df_uop.sort_values('charttime').groupby('stay_id'):
    uop_dict[stay_id] = {
        'times': group['charttime'].values,
        'vals': group['value'].values
    }

vitals_defaults = {'heart_rate': 80.0, 'resp_rate': 15.0, 'spo2': 98.0, 'sbp': 120.0, 'dbp': 80.0, 'mbp': 80.0, 'temp': 98.6}
vitals_cols_to_ids = {col: [k for k, v in vitals_mapping.items() if v == col] for col in vitals_mapping.values()}
labs_defaults = {
    'glucose': 100.0, 'potassium': 4.0, 'sodium': 140.0, 'creatinine': 0.8,
    'chloride': 100.0, 'bun': 15.0, 'hematocrit': 40.0, 'bicarbonate': 24.0,
    'platelets': 200.0, 'hemoglobin': 13.0, 'wbc': 7.0, 'bilirubin': 0.5, 'lactate': 1.0
}
labs_cols_to_ids = {col: [k for k, v in labs_mapping.items() if v == col] for col in set(labs_mapping.values())}

print("Starting hourly grid construction...")
for idx_stay, (_, stay) in enumerate(df_icustays.iterrows()):
    if idx_stay % 5000 == 0:
        print(f"Processed {idx_stay}/{len(df_icustays)} ICU stays...")
        
    stay_id = stay['stay_id']
    subject_id = stay['subject_id']
    hadm_id = stay['hadm_id']
    
    pat = pat_dict.get(subject_id, None)
    if pat is None:
        continue
    age = pat['anchor_age']
    gender = pat['gender']
    
    intime = pd.to_datetime(stay['intime'])
    outtime = pd.to_datetime(stay['outtime'])
    
    if pd.isna(intime) or pd.isna(outtime):
        continue
    
    start_hour = intime.replace(minute=0, second=0, microsecond=0)
    end_hour = outtime.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    grid_times = pd.date_range(start=start_hour, end=end_hour, freq='h')
    
    if len(grid_times) < 6:
        continue
        
    # Vitals processing
    vitals_dict = {}
    icu_stay_data = icu_dict.get(stay_id, {})
    for col_name, itemids in vitals_cols_to_ids.items():
        meas_list = []
        for itemid in itemids:
            meas = icu_stay_data.get(itemid, None)
            if meas is not None:
                meas_list.append(meas)
        
        if len(meas_list) > 0:
            all_times = np.concatenate([m['times'] for m in meas_list])
            all_vals = np.concatenate([m['vals'] for m in meas_list])
            sort_idx = np.argsort(all_times)
            var_times = all_times[sort_idx]
            var_vals = all_vals[sort_idx]
            
            vals_grid = np.full(len(grid_times), np.nan)
            var_idx = 0
            current_val = np.nan
            for i, t in enumerate(grid_times):
                while var_idx < len(var_times) and var_times[var_idx] <= t:
                    current_val = var_vals[var_idx]
                    var_idx += 1
                vals_grid[i] = current_val
            vitals_dict[col_name] = vals_grid
        else:
            vitals_dict[col_name] = np.full(len(grid_times), np.nan)
            
    vitals_dict['sbp'] = np.where(~np.isnan(vitals_dict['sbp_art']), vitals_dict['sbp_art'], vitals_dict['sbp_ni'])
    vitals_dict['dbp'] = np.where(~np.isnan(vitals_dict['dbp_art']), vitals_dict['dbp_art'], vitals_dict['dbp_ni'])
    vitals_dict['mbp'] = np.where(~np.isnan(vitals_dict['mbp_art']), vitals_dict['mbp_art'], vitals_dict['mbp_ni'])
    
    temp_c = vitals_dict['temp_c']
    temp_f = vitals_dict['temp_f']
    temp_combined = np.full(len(grid_times), np.nan)
    for i in range(len(grid_times)):
        tc = temp_c[i]
        tf = temp_f[i]
        if not np.isnan(tc):
            temp_combined[i] = tc * 1.8 + 32.0
        elif not np.isnan(tf):
            temp_combined[i] = tf
    vitals_dict['temp'] = temp_combined
    
    for col in ['heart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp', 'temp']:
        s = pd.Series(vitals_dict[col]).ffill().bfill().fillna(vitals_defaults[col])
        vitals_dict[col] = s.values
        
    # Labs processing
    labs_dict = {}
    hosp_subj_data = hosp_dict.get(subject_id, {})
    for col_name, itemids in labs_cols_to_ids.items():
        meas_list = []
        for itemid in itemids:
            meas = hosp_subj_data.get(itemid, None)
            if meas is not None:
                mask = (meas['times'] >= intime - pd.Timedelta(hours=24)) & (meas['times'] <= outtime)
                if np.any(mask):
                    meas_list.append({
                        'times': meas['times'][mask],
                        'vals': meas['vals'][mask]
                    })
        
        if len(meas_list) > 0:
            all_times = np.concatenate([m['times'] for m in meas_list])
            all_vals = np.concatenate([m['vals'] for m in meas_list])
            sort_idx = np.argsort(all_times)
            var_times = all_times[sort_idx]
            var_vals = all_vals[sort_idx]
            
            vals_grid = np.full(len(grid_times), np.nan)
            var_idx = 0
            current_val = np.nan
            for i, t in enumerate(grid_times):
                while var_idx < len(var_times) and var_times[var_idx] <= t:
                    current_val = var_vals[var_idx]
                    var_idx += 1
                vals_grid[i] = current_val
            labs_dict[col_name] = vals_grid
        else:
            labs_dict[col_name] = np.full(len(grid_times), np.nan)
            
    for col in labs_defaults.keys():
        s = pd.Series(labs_dict[col]).ffill().bfill().fillna(labs_defaults[col])
        labs_dict[col] = s.values
        
    # SOFA Respiration
    sofa_resp = np.zeros(len(grid_times))
    pao2_meas = hosp_subj_data.get(50821, None)
    pao2_vals = np.full(len(grid_times), np.nan)
    if pao2_meas is not None:
        mask = (pao2_meas['times'] >= intime - pd.Timedelta(hours=24)) & (pao2_meas['times'] <= outtime)
        if np.any(mask):
            var_times = pao2_meas['times'][mask]
            var_vals = pao2_meas['vals'][mask]
            var_idx = 0
            current_val = np.nan
            for i, t in enumerate(grid_times):
                while var_idx < len(var_times) and var_times[var_idx] <= t:
                    current_val = var_vals[var_idx]
                    var_idx += 1
                pao2_vals[i] = current_val
                
    spo2_meas = icu_stay_data.get(220277, None)
    spo2_vals = np.full(len(grid_times), 98.0)
    if spo2_meas is not None:
        var_times = spo2_meas['times']
        var_vals = spo2_meas['vals']
        var_idx = 0
        current_val = 98.0
        for i, t in enumerate(grid_times):
            while var_idx < len(var_times) and var_times[var_idx] <= t:
                current_val = var_vals[var_idx]
                var_idx += 1
            spo2_vals[i] = current_val
            
    fio2_meas = icu_stay_data.get(223835, None)
    fio2_vals = np.full(len(grid_times), 0.21)
    if fio2_meas is not None:
        var_times = fio2_meas['times']
        var_vals = fio2_meas['vals']
        var_idx = 0
        current_val = 0.21
        for i, t in enumerate(grid_times):
            while var_idx < len(var_times) and var_times[var_idx] <= t:
                val = var_vals[var_idx]
                if val > 1.0:
                    val = val / 100.0
                current_val = max(0.21, min(1.0, val))
                var_idx += 1
            fio2_vals[i] = current_val
            
    for i in range(len(grid_times)):
        current_pao2 = pao2_vals[i]
        current_spo2 = spo2_vals[i]
        current_fio2 = fio2_vals[i]
        pf_ratio = current_pao2 / current_fio2 if not np.isnan(current_pao2) else np.nan
        sf_ratio = current_spo2 / current_fio2
        if not np.isnan(pf_ratio):
            if pf_ratio < 100:
                sofa_resp[i] = 4
            elif pf_ratio < 200:
                sofa_resp[i] = 3
            elif pf_ratio < 300:
                sofa_resp[i] = 2
            elif pf_ratio < 400:
                sofa_resp[i] = 1
        else:
            if sf_ratio < 150:
                sofa_resp[i] = 4
            elif sf_ratio < 235:
                sofa_resp[i] = 3
            elif sf_ratio < 315:
                sofa_resp[i] = 2
            elif sf_ratio < 400:
                sofa_resp[i] = 1
                
    # SOFA Coagulation
    sofa_coag = np.zeros(len(grid_times))
    plat_meas_list = []
    for itemid in [51265, 51704]:
        meas = hosp_subj_data.get(itemid, None)
        if meas is not None:
            plat_meas_list.append(meas)
    if len(plat_meas_list) > 0:
        all_times = np.concatenate([m['times'] for m in plat_meas_list])
        all_vals = np.concatenate([m['vals'] for m in plat_meas_list])
        sort_idx = np.argsort(all_times)
        plat_times = all_times[sort_idx]
        plat_vals = all_vals[sort_idx]
        
        plat_idx = 0
        current_plat = 200.0
        for i, t in enumerate(grid_times):
            while plat_idx < len(plat_times) and plat_times[plat_idx] <= t:
                current_plat = plat_vals[plat_idx]
                plat_idx += 1
            if current_plat < 20:
                sofa_coag[i] = 4
            elif current_plat < 50:
                sofa_coag[i] = 3
            elif current_plat < 100:
                sofa_coag[i] = 2
            elif current_plat < 150:
                sofa_coag[i] = 1
                
    # SOFA Liver
    sofa_liver = np.zeros(len(grid_times))
    bili_meas = hosp_subj_data.get(50885, None)
    if bili_meas is not None:
        bili_times = bili_meas['times']
        bili_vals = bili_meas['vals']
        bili_idx = 0
        current_bili = 0.5
        for i, t in enumerate(grid_times):
            while bili_idx < len(bili_times) and bili_times[bili_idx] <= t:
                current_bili = bili_vals[bili_idx]
                bili_idx += 1
            if current_bili >= 12.0:
                sofa_liver[i] = 4
            elif current_bili >= 6.0:
                sofa_liver[i] = 3
            elif current_bili >= 2.0:
                sofa_liver[i] = 2
            elif current_bili >= 1.2:
                sofa_liver[i] = 1
                
    # SOFA Cardiovascular
    sofa_cardio = np.zeros(len(grid_times))
    map_meas_list = []
    for itemid in [220052, 220181]:
        meas = icu_stay_data.get(itemid, None)
        if meas is not None:
            map_meas_list.append(meas)
            
    map_times = np.array([])
    map_vals = np.array([])
    if len(map_meas_list) > 0:
        all_times = np.concatenate([m['times'] for m in map_meas_list])
        all_vals = np.concatenate([m['vals'] for m in map_meas_list])
        sort_idx = np.argsort(all_times)
        map_times = all_times[sort_idx]
        map_vals = all_vals[sort_idx]
        
    active_vasos = vaso_dict.get(stay_id, [])
    map_idx = 0
    current_map = 80.0
    for i, t in enumerate(grid_times):
        if len(map_times) > 0:
            while map_idx < len(map_times) and map_times[map_idx] <= t:
                val = map_vals[map_idx]
                if not np.isnan(val) and 30 < val < 200:
                    current_map = val
                map_idx += 1
                
        t_active_vasos = [v for v in active_vasos if v['start'] <= t <= v['end']]
        if len(t_active_vasos) > 0:
            max_score = 2
            for vaso in t_active_vasos:
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
            sofa_cardio[i] = max_score
        else:
            if current_map < 70.0:
                sofa_cardio[i] = 1
                
    # SOFA CNS
    sofa_cns = np.zeros(len(grid_times))
    gcs_eye = np.full(len(grid_times), 4.0)
    gcs_verbal = np.full(len(grid_times), 5.0)
    gcs_motor = np.full(len(grid_times), 6.0)
    
    eye_meas = icu_stay_data.get(220739, None)
    if eye_meas is not None:
        times, vals = eye_meas['times'], eye_meas['vals']
        idx = 0
        cur = 4.0
        for i, t in enumerate(grid_times):
            while idx < len(times) and times[idx] <= t:
                cur = vals[idx]
                idx += 1
            gcs_eye[i] = cur
            
    verbal_meas = icu_stay_data.get(223900, None)
    if verbal_meas is not None:
        times, vals = verbal_meas['times'], verbal_meas['vals']
        idx = 0
        cur = 5.0
        for i, t in enumerate(grid_times):
            while idx < len(times) and times[idx] <= t:
                cur = vals[idx]
                idx += 1
            gcs_verbal[i] = cur
            
    motor_meas = icu_stay_data.get(223901, None)
    if motor_meas is not None:
        times, vals = motor_meas['times'], motor_meas['vals']
        idx = 0
        cur = 6.0
        for i, t in enumerate(grid_times):
            while idx < len(times) and times[idx] <= t:
                cur = vals[idx]
                idx += 1
            gcs_motor[i] = cur
            
    for i in range(len(grid_times)):
        current_gcs = gcs_eye[i] + gcs_verbal[i] + gcs_motor[i]
        if current_gcs < 6:
            sofa_cns[i] = 4
        elif current_gcs < 10:
            sofa_cns[i] = 3
        elif current_gcs < 13:
            sofa_cns[i] = 2
        elif current_gcs < 15:
            sofa_cns[i] = 1
            
    # SOFA Renal
    creat_scores = np.zeros(len(grid_times))
    creat_meas_list = []
    for itemid in [50912, 52024]:
        meas = hosp_subj_data.get(itemid, None)
        if meas is not None:
            creat_meas_list.append(meas)
    if len(creat_meas_list) > 0:
        all_times = np.concatenate([m['times'] for m in creat_meas_list])
        all_vals = np.concatenate([m['vals'] for m in creat_meas_list])
        sort_idx = np.argsort(all_times)
        creat_times = all_times[sort_idx]
        creat_vals = all_vals[sort_idx]
        
        creat_idx = 0
        current_creat = 0.8
        for i, t in enumerate(grid_times):
            while creat_idx < len(creat_times) and creat_times[creat_idx] <= t:
                current_creat = creat_vals[creat_idx]
                creat_idx += 1
            if current_creat >= 5.0:
                creat_scores[i] = 4
            elif current_creat >= 3.5:
                creat_scores[i] = 3
            elif current_creat >= 2.0:
                creat_scores[i] = 2
            elif current_creat >= 1.2:
                creat_scores[i] = 1
                
    uop_scores = np.zeros(len(grid_times))
    uop_meas = uop_dict.get(stay_id, None)
    if uop_meas is not None:
        uop_times = uop_meas['times']
        uop_vals = uop_meas['vals']
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
    sofa_renal = np.maximum(creat_scores, uop_scores)
    
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

print("Building feature matrix and target labels (lookback=6h)...")
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
    baseline_sofa = sofa_vals[0] if len(sofa_vals) > 0 else np.nan
    for idx in range(len(df_win)):
        current_sofa = sofa_vals[idx]
        current_time = times[idx]
        if not np.isnan(baseline_sofa) and current_sofa - baseline_sofa >= 2:
            onset_time = current_time
            break
    if onset_time is not None:
        sepsis_onsets[stay_id] = onset_time

print(f"Total stays with Sepsis-3 onset: {len(sepsis_onsets)} out of {len(grouped_stays)}")

vitals_cols = ['heart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp', 'temp']
labs_cols = ['glucose', 'potassium', 'sodium', 'creatinine', 'chloride', 'bun', 
             'hematocrit', 'bicarbonate', 'platelets', 'hemoglobin', 'wbc', 'bilirubin', 'lactate']

feature_rows = []
for stay_id, df_stay in grouped_stays:
    df_stay = df_stay.sort_values('time').reset_index(drop=True)
    subject_id = df_stay.iloc[0]['subject_id']
    age = df_stay.iloc[0]['age']
    gender = df_stay.iloc[0]['gender']
    onset_time = sepsis_onsets.get(stay_id, None)
    
    for idx in range(5, len(df_stay)):
        current_time = df_stay.iloc[idx]['time']
        if onset_time is not None and current_time >= onset_time:
            continue
        df_lookback = df_stay.iloc[idx - 5:idx + 1]
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
        
        for col in vitals_cols:
            vals = df_lookback[col].values
            row_features[f'{col}_mean'] = np.mean(vals)
            row_features[f'{col}_std'] = np.std(vals)
            row_features[f'{col}_min'] = np.min(vals)
            row_features[f'{col}_max'] = np.max(vals)
            row_features[f'{col}_last'] = vals[-1]
            
            n = len(vals)
            x = np.arange(n)
            denom = n * np.sum(x**2) - np.sum(x)**2
            if denom == 0:
                slope = 0.0
            else:
                num = n * np.sum(x * vals) - np.sum(x) * np.sum(vals)
                slope = num / denom
            row_features[f'{col}_slope'] = slope
            
        for col in labs_cols:
            vals = df_lookback[col].values
            row_features[f'{col}_last'] = vals[-1]
            
        if onset_time is not None:
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

df_features.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)

stats = {
    'total_patients': int(df_patients['subject_id'].nunique()),
    'total_stays': int(df_icustays['stay_id'].nunique()),
    'stays_with_sepsis': int(len(sepsis_onsets)),
    'total_rows': int(len(df_features)),
    'positive_rows': int(df_features['label'].sum()),
    'positive_rate': float(df_features['label'].mean()),
}
with open(os.path.join(PROCESSED_DIR, 'stats.json'), 'w', encoding='utf-8') as handle:
    json.dump(stats, handle, indent=4)
print("Saved preprocessing statistics to Data/processed/stats.json")

print("Preprocessing completed successfully!")
print("Generating sepsis_sequential.npz for RNN training...")
seq_features = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temp', 'spo2', 'glucose', 'sodium', 'creatinine', 'wbc']
try:
    df_static = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    stay_level = (
        df_static.groupby('stay_id', as_index=False)
        .agg({
            'label': 'max',
            'age': 'first',
            'gender': 'first'
        })
    )
    stay_level = stay_level.sort_values('stay_id').reset_index(drop=True)
    target_stays = stay_level['stay_id'].astype(int).values
    y_static = stay_level['label'].astype(int).values

    X_static_all = stay_level[['age', 'gender']].copy()
    X_static_all['gender'] = pd.to_numeric(X_static_all['gender'], errors='coerce').fillna(0).astype(int)
    X_static_all['age'] = pd.to_numeric(X_static_all['age'], errors='coerce').fillna(0)
    X_static_array = X_static_all.values.astype(np.float32)

    N = len(target_stays)
    X_seq_array = np.full((N, 24, len(seq_features)), np.nan, dtype=np.float32)

    grouped_grid = df_grid.groupby('stay_id')
    for i, sid in enumerate(target_stays):
        if sid in grouped_grid.groups:
            group = grouped_grid.get_group(sid).sort_values('time')
            vals = group[seq_features].values[:24]
            actual_len = min(24, len(vals))
            X_seq_array[i, :actual_len, :] = vals

    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "sepsis_sequential.npz"),
        X_seq=X_seq_array,
        X_static=X_static_array,
        y=y_static,
        stay_ids=target_stays
    )
    print("Saved sepsis_sequential.npz successfully!")
except Exception as e:
    print("Could not generate sepsis_sequential.npz:", e)

print(f"total_patients: {df_patients['subject_id'].nunique()}")
print(f"total_stays: {df_icustays['stay_id'].nunique()}")
print(f"stays_in_grid: {df_grid['stay_id'].nunique()}")
print(f"stays_with_sepsis: {len(sepsis_onsets)}")
print(f"total_rows_features: {len(df_features)}")
print(f"positive_rows: {df_features['label'].sum()}")
print(f"negative_rows: {len(df_features) - df_features['label'].sum()}")