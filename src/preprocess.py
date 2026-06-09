import os
import re
import numpy as np
import pandas as pd

DATA_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\mimic-iv-clinical-database-demo-2.2"
PROCESSED_DIR = r"D:\Internship2026\Infectious-Disease-Triage\Data\processed"
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
df_patients = pd.read_csv(os.path.join(DATA_DIR, "hosp", "patients.csv.gz"))
print("Loading admissions.csv.gz...")
df_admissions = pd.read_csv(os.path.join(DATA_DIR, "hosp", "admissions.csv.gz"))
print("Loading icustays.csv.gz...")
df_icustays = pd.read_csv(os.path.join(DATA_DIR, "icu", "icustays.csv.gz"))
print("Loading microbiologyevents.csv.gz...")
df_micro = pd.read_csv(os.path.join(DATA_DIR, "hosp", "microbiologyevents.csv.gz"))
print("Loading prescriptions.csv.gz...")
df_rx = pd.read_csv(os.path.join(DATA_DIR, "hosp", "prescriptions.csv.gz"))
print("Loading chartevents.csv.gz...")
df_chartevents = pd.read_csv(os.path.join(DATA_DIR, "icu", "chartevents.csv.gz"))
print("Loading labevents.csv.gz...")
df_labevents = pd.read_csv(os.path.join(DATA_DIR, "hosp", "labevents.csv.gz"))
print("Loading inputevents.csv.gz...")
df_vaso = pd.read_csv(os.path.join(DATA_DIR, "icu", "inputevents.csv.gz"))
print("Loading outputevents.csv.gz...")
df_uop = pd.read_csv(os.path.join(DATA_DIR, "icu", "outputevents.csv.gz"))

vaso_itemids = [221906, 221289, 229617, 221662, 221653, 222315]
df_vaso = df_vaso[df_vaso['itemid'].isin(vaso_itemids)].copy()
df_vaso['starttime'] = pd.to_datetime(df_vaso['starttime'])
df_vaso['endtime'] = pd.to_datetime(df_vaso['endtime'])

uop_itemids = [226559, 226566, 226627, 226631]
df_uop = df_uop[df_uop['itemid'].isin(uop_itemids)].copy()
df_uop['charttime'] = pd.to_datetime(df_uop['charttime'])

print("Identifying suspected infections at hadm_id level...")
df_micro_temp = df_micro.copy()
df_micro_temp['charttime'] = pd.to_datetime(df_micro_temp['charttime'].fillna(df_micro_temp['chartdate']))

df_rx_temp = df_rx.copy()
df_rx_temp['starttime'] = pd.to_datetime(df_rx_temp['starttime'])

pattern = '|'.join(ABX_PATTERNS)
df_rx_abx = df_rx_temp[df_rx_temp['drug'].str.contains(pattern, case=False, na=False)].copy()
for non_ab in NON_ABX:
    df_rx_abx = df_rx_abx[~df_rx_abx['drug'].str.contains(non_ab, case=False, na=False)]

si_records = []
micro_grouped = df_micro_temp.groupby('hadm_id')
rx_grouped = df_rx_abx.groupby('hadm_id')

for hadm_id, micro_group in micro_grouped:
    if pd.isna(hadm_id) or hadm_id not in rx_grouped.groups:
        continue
    rx_group = rx_grouped.get_group(hadm_id)
    
    for _, micro_row in micro_group.iterrows():
        t_culture = micro_row['charttime']
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
    df_si = df_si.sort_values(by='t_si').groupby('hadm_id').first().reset_index()
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
    
    if len(grid_times) < 6:
        continue
        
    df_stay_icu = df_icu_data[df_icu_data['stay_id'] == stay_id].copy()
    df_stay_hosp = df_hosp_data[
        (df_hosp_data['subject_id'] == subject_id) &
        (df_hosp_data['charttime'] >= intime - pd.Timedelta(hours=24)) &
        (df_hosp_data['charttime'] <= outtime)
    ].copy()
    
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

    df_pao2 = df_stay_hosp[df_stay_hosp['itemid'] == 50821].copy()
    df_resp_icu = df_stay_icu[df_stay_icu['itemid'].isin([220277, 223835])].copy()
    df_resp = pd.concat([df_pao2[['charttime', 'itemid', 'valuenum']], df_resp_icu[['charttime', 'itemid', 'valuenum']]])
    
    sofa_resp = np.zeros(len(grid_times))
    if not df_resp.empty:
        df_resp = df_resp.sort_values('charttime')
        df_pivot_resp = df_resp.pivot_table(index='charttime', columns='itemid', values='valuenum', aggfunc='last').reset_index()
        if 223835 in df_pivot_resp.columns:
            df_pivot_resp[223835] = df_pivot_resp[223835].apply(lambda x: x / 100.0 if x > 1.0 else x)
            df_pivot_resp[223835] = df_pivot_resp[223835].clip(0.21, 1.0)
        else:
            df_pivot_resp[223835] = np.nan
        df_pivot_resp = df_pivot_resp.ffill().fillna({223835: 0.21})
        if 220277 not in df_pivot_resp.columns:
            df_pivot_resp[220277] = np.nan
        df_pivot_resp[220277] = df_pivot_resp[220277].fillna(98.0)
        if 50821 not in df_pivot_resp.columns:
            df_pivot_resp[50821] = np.nan
        resp_times = df_pivot_resp['charttime'].values
        resp_idx = 0
        current_pao2 = np.nan
        current_spo2 = 98.0
        current_fio2 = 0.21
        for i, t in enumerate(grid_times):
            while resp_idx < len(df_pivot_resp) and resp_times[resp_idx] <= t:
                row_resp = df_pivot_resp.iloc[resp_idx]
                current_pao2 = row_resp[50821] if 50821 in row_resp else np.nan
                current_spo2 = row_resp[220277] if 220277 in row_resp else 98.0
                current_fio2 = row_resp[223835] if 223835 in row_resp else 0.21
                resp_idx += 1
            pf_ratio = current_pao2 / current_fio2 if not pd.isna(current_pao2) else np.nan
            sf_ratio = current_spo2 / current_fio2
            if not pd.isna(pf_ratio):
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
        
    df_plat = df_stay_hosp[df_stay_hosp['itemid'].isin([51265, 51704])].copy()
    sofa_coag = np.zeros(len(grid_times))
    if not df_plat.empty:
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
                sofa_coag[i] = 4
            elif current_plat < 50:
                sofa_coag[i] = 3
            elif current_plat < 100:
                sofa_coag[i] = 2
            elif current_plat < 150:
                sofa_coag[i] = 1
        
    df_bili = df_stay_hosp[df_stay_hosp['itemid'] == 50885].copy()
    sofa_liver = np.zeros(len(grid_times))
    if not df_bili.empty:
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
                sofa_liver[i] = 4
            elif current_bili >= 6.0:
                sofa_liver[i] = 3
            elif current_bili >= 2.0:
                sofa_liver[i] = 2
            elif current_bili >= 1.2:
                sofa_liver[i] = 1
        
    df_map = df_stay_icu[df_stay_icu['itemid'].isin([220052, 220181])].copy()
    df_stay_vaso = df_vaso[df_vaso['stay_id'] == stay_id].copy()
    sofa_cardio = np.zeros(len(grid_times))
    vaso_active = []
    if not df_stay_vaso.empty:
        for _, row in df_stay_vaso.iterrows():
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
            sofa_cardio[i] = max_score
        else:
            if current_map < 70.0:
                sofa_cardio[i] = 1
        
    df_gcs = df_stay_icu[df_stay_icu['itemid'].isin(gcs_itemids)].copy()
    sofa_cns = np.zeros(len(grid_times))
    if not df_gcs.empty:
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
                sofa_cns[i] = 4
            elif current_gcs < 10:
                sofa_cns[i] = 3
            elif current_gcs < 13:
                sofa_cns[i] = 2
            elif current_gcs < 15:
                sofa_cns[i] = 1
        
    df_creat = df_stay_hosp[df_stay_hosp['itemid'].isin([50912, 52024])].copy()
    df_stay_uop = df_uop[df_uop['stay_id'] == stay_id].copy()
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
    uop_scores = np.zeros(len(grid_times))
    if not df_stay_uop.empty:
        df_stay_uop = df_stay_uop.sort_values('charttime')
        uop_times = df_stay_uop['charttime'].values
        uop_vals = df_stay_uop['value'].values
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

print("Preprocessing completed successfully!")
print(f"total_patients: {df_patients['subject_id'].nunique()}")
print(f"total_stays: {df_icustays['stay_id'].nunique()}")
print(f"stays_in_grid: {df_grid['stay_id'].nunique()}")
print(f"stays_with_sepsis: {len(sepsis_onsets)}")
print(f"total_rows_features: {len(df_features)}")
print(f"positive_rows: {df_features['label'].sum()}")
print(f"negative_rows: {len(df_features) - df_features['label'].sum()}")
