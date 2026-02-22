import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import plotly.graph_objects as go
from scipy.stats import norm as scipy_norm
import io

st.set_page_config(page_title="Stuff+ Arsenal Builder", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>div[data-testid='stDecoration']{display:none}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TrackMan CSV column mapping (handles common naming variations)
# ------------------------------------------------------------------
TM_COL_MAP = {
    'pitcher':          ['Pitcher', 'PitcherName', 'pitcher', 'pitcher_name'],
    'pitch_type':       ['TaggedPitchType', 'AutoPitchType', 'pitch_type', 'PitchType'],
    'rel_speed':        ['RelSpeed', 'Velocity', 'rel_speed', 'velo', 'Speed'],
    'induced_vert':     ['InducedVertBreak', 'InducedVBreak', 'induced_vert_break', 'iVB', 'InducedVB'],
    'horz_break':       ['HorzBreak', 'HorizontalBreak', 'horz_break', 'HB'],
    'spin_rate':        ['SpinRate', 'spin_rate', 'Spin', 'SpinRPM'],
    'extension':        ['Extension', 'ReleaseExtension', 'extension', 'Ext'],
    'rel_height':       ['RelHeight', 'ReleaseHeight', 'rel_height', 'RelZ'],
    'rel_side':         ['RelSide', 'ReleaseSide', 'rel_side', 'RelX'],
    'pitcher_hand':     ['PitcherThrows', 'ThrowingHand', 'pitcher_hand', 'Throws', 'Hand'],
    'plate_loc_side':   ['PlateLocSide', 'plate_loc_side', 'PlateX', 'px'],
    'plate_loc_height': ['PlateLocHeight', 'plate_loc_height', 'PlateZ', 'pz'],
}

PITCH_TYPE_NORMALIZE = {
    'Four-Seam': 'Fastball', 'FourSeam': 'Fastball', '4-Seam': 'Fastball', '4S': 'Fastball',
    'FF': 'Fastball', 'FA': 'Fastball', 'Fastball': 'Fastball',
    'Two-Seam': 'Sinker', 'TwoSeam': 'Sinker', '2S': 'Sinker', 'SI': 'Sinker',
    'Sinker': 'Sinker', 'SN': 'Sinker',
    'Slider': 'Slider', 'SL': 'Slider', 'Sweeper': 'Slider',
    'Curveball': 'Curveball', 'CU': 'Curveball', 'CB': 'Curveball', 'KC': 'Curveball',
    'KnuckleCurve': 'Curveball', '12-6': 'Curveball',
    'ChangeUp': 'ChangeUp', 'Changeup': 'ChangeUp', 'CH': 'ChangeUp',
    'Change-Up': 'ChangeUp', 'CHS': 'ChangeUp',
    'Cutter': 'Cutter', 'CT': 'Cutter', 'FC': 'Cutter',
    'Splitter': 'Splitter', 'SP': 'Splitter', 'FS': 'Splitter', 'Split': 'Splitter',
}

VALID_PITCH_TYPES = ["Fastball", "Sinker", "Slider", "Curveball", "ChangeUp", "Cutter", "Splitter"]

def find_col(df, key):
    for name in TM_COL_MAP.get(key, []):
        if name in df.columns:
            return name
    return None

def normalize_pitch_type(pt):
    if pd.isna(pt): return None
    pt_str = str(pt).strip()
    return PITCH_TYPE_NORMALIZE.get(pt_str, pt_str if pt_str in VALID_PITCH_TYPES else None)

def parse_trackman_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    col_results = {k: find_col(df, k) for k in TM_COL_MAP}
    missing_critical = [k for k in ['pitcher', 'pitch_type', 'rel_speed', 'induced_vert', 'horz_break', 'spin_rate']
                        if col_results[k] is None]
    if missing_critical:
        return None, f"Missing required columns: {missing_critical}. Found columns: {list(df.columns[:20])}"

    out = pd.DataFrame()
    for key, col in col_results.items():
        if col: out[key] = df[col]
        else:   out[key] = np.nan

    out['pitch_type_raw'] = out['pitch_type'].copy()
    out['pitch_type']     = out['pitch_type'].apply(normalize_pitch_type)

    numeric_cols = ['rel_speed', 'induced_vert', 'horz_break', 'spin_rate']
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors='coerce')
    out = out.dropna(subset=numeric_cols + ['pitcher', 'pitch_type'])
    out = out[out['pitch_type'].isin(VALID_PITCH_TYPES)]

    out['extension']  = pd.to_numeric(out['extension'],  errors='coerce').fillna(6.0)
    out['rel_height'] = pd.to_numeric(out['rel_height'], errors='coerce').fillna(5.5)
    out['rel_side']   = pd.to_numeric(out['rel_side'],   errors='coerce').fillna(2.0)

    if col_results['pitcher_hand']:
        out['pitcher_hand'] = df[col_results['pitcher_hand']].map(
            lambda x: 'Right' if str(x).strip() in ['R', 'Right', 'RHP', 'right'] else 'Left')
    else:
        out['pitcher_hand'] = out['rel_side'].apply(lambda x: 'Right' if x > 0 else 'Left')

    return out, None


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open('stuff_event_models.pkl', 'rb') as f:
        event_models = pickle.load(f)
    final_model = lgb.Booster(model_file='stuff_final_model.txt')
    with open('stuff_normalization_params.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    with open('stuff_model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('pitching_event_models.pkl', 'rb') as f:
        pitching_event_models = pickle.load(f)
    pitching_final_model = lgb.Booster(model_file='pitching_final_model.txt')
    with open('location_sd_params.pkl', 'rb') as f:
        location_sd_params = pickle.load(f)
    return event_models, final_model, norm_params, metadata, pitching_event_models, pitching_final_model, location_sd_params

event_models, final_model, norm_params, metadata, pitching_event_models, pitching_final_model, location_sd_params = load_models()
stuff_features    = metadata['stuff_features']
pitching_features = metadata['pitching_features']
pitch_type_map    = metadata['pitch_type_map']
fb_types          = metadata['fb_types']
labels            = metadata['labels']

def fmt(x):
    if pd.isna(x): return x
    try:
        v = float(x)
        return int(v) if v == int(v) else round(v, 1)
    except:
        return x

PITCH_DEFAULTS = {
    "Fastball": {"velo": 94.0, "ivb": 16.0, "hb": 8.0,  "spin": 2300},
    "Sinker":   {"velo": 93.0, "ivb": 8.0,  "hb": 14.0, "spin": 2150},
    "Slider":   {"velo": 85.0, "ivb": 1.0,  "hb": -5.0, "spin": 2450},
    "Curveball":{"velo": 78.0, "ivb": -8.0, "hb": 6.0,  "spin": 2600},
    "ChangeUp": {"velo": 84.0, "ivb": 10.0, "hb": 12.0, "spin": 1700},
    "Cutter":   {"velo": 89.0, "ivb": 10.0, "hb": -2.0, "spin": 2400},
    "Splitter": {"velo": 86.0, "ivb": 4.0,  "hb": 10.0, "spin": 1400},
}

if 'pitches'       not in st.session_state: st.session_state.pitches = []
if 'active_tab'    not in st.session_state: st.session_state.active_tab = 0
if 'tm_data'       not in st.session_state: st.session_state.tm_data = None
if 'input_mode'    not in st.session_state: st.session_state.input_mode = "Arsenal Builder"

# ------------------------------------------------------------------
# Helpers: Score computation
# ------------------------------------------------------------------
def compute_stuff_scores(pitches_list, pitcher_is_righty):
    pitch_counts = {}
    for p in pitches_list:
        if p['pitch_type'] in fb_types:
            pitch_counts[p['pitch_type']] = pitch_counts.get(p['pitch_type'], 0) + 1

    if pitch_counts:
        primary_fb = max(pitch_counts, key=pitch_counts.get)
        fb_pitches = [p for p in pitches_list if p['pitch_type'] == primary_fb]
    else:
        fb_pitches = [pitches_list[0]]

    avg_fb_vel  = np.mean([p['rel_speed']    for p in fb_pitches])
    avg_fb_vert = np.mean([p['induced_vert'] for p in fb_pitches])
    avg_fb_horz = np.mean([p['horz_break']   for p in fb_pitches])
    avg_fb_spin = np.mean([p['spin_rate']     for p in fb_pitches])

    rows = []
    for p in pitches_list:
        rows.append({
            'RelSpeed':          p['rel_speed'],
            'InducedVertBreak':  p['induced_vert'],
            'HorzBreak':         p['horz_break'],
            'SpinRate':          p['spin_rate'],
            'Extension':         p['extension'],
            'RelHeight':         p['rel_height'],
            'RelSide':           p['rel_side'],
            'TaggedPitchTypeNum':pitch_type_map[p['pitch_type']],
            'pitcher_is_righty': pitcher_is_righty,
            'diff_fb_vel':       avg_fb_vel  - p['rel_speed'],
            'diff_fb_vert':      avg_fb_vert - p['induced_vert'],
            'diff_fb_horz':      avg_fb_horz - p['horz_break'],
            'diff_spinrate':     avg_fb_spin - p['spin_rate'],
        })
    df = pd.DataFrame(rows)

    for label in labels:
        df[f'sp_{label}'] = event_models[label].predict(
            df[stuff_features].values, num_iteration=event_models[label].best_iteration)

    spf = [c for c in df.columns if c.startswith('sp_')]
    df['stuff_lw']         = final_model.predict(df[spf].values)
    df['StuffPlus']        = 100 + (norm_params['mu_stuff']       - df['stuff_lw'])                / norm_params['sigma_stuff']          * 10
    df['WhiffPlus']        = 100 + (df['sp_StrikeSwinging']       - norm_params['mu_whiff'])        / norm_params['sigma_whiff']          * 10
    df['CalledStrikePlus'] = 100 + (df['sp_StrikeCalled']         - norm_params['mu_called_strike']) / norm_params['sigma_called_strike']  * 10
    df['wOBACON']          = (
        (df['sp_Single']*1 + df['sp_Double']*2 + df['sp_Triple']*3 + df['sp_HomeRun']*4) /
        (df['sp_Single'] + df['sp_Double'] + df['sp_Triple'] + df['sp_HomeRun'] + 0.0001))
    df['wOBACONPlus']      = 100 - (df['wOBACON'] - norm_params['mu_wobacon']) / norm_params['sigma_wobacon'] * 10
    df['PitchType']        = [p['pitch_type'] for p in pitches_list]
    return df, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin


@st.cache_data
def compute_zone_heatmap(pitch_row: dict, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin,
                          pitcher_is_righty, _pitching_event_models, _pitching_final_model,
                          norm_params, pitching_features, pitch_type_map, labels, grid_n=25):
    side_vals   = np.linspace(-3.0, 3.0, grid_n)
    height_vals = np.linspace(-0.5, 5.5, grid_n)
    sg, hg = np.meshgrid(side_vals, height_vals)
    n = sg.size

    data = {
        'RelSpeed':          np.full(n, pitch_row['rel_speed']),
        'InducedVertBreak':  np.full(n, pitch_row['induced_vert']),
        'HorzBreak':         np.full(n, pitch_row['horz_break']),
        'SpinRate':          np.full(n, pitch_row['spin_rate']),
        'Extension':         np.full(n, pitch_row['extension']),
        'RelHeight':         np.full(n, pitch_row['rel_height']),
        'RelSide':           np.full(n, pitch_row['rel_side']),
        'TaggedPitchTypeNum':np.full(n, pitch_type_map[pitch_row['pitch_type']]),
        'pitcher_is_righty': np.full(n, pitcher_is_righty),
        'diff_fb_vel':       np.full(n, avg_fb_vel  - pitch_row['rel_speed']),
        'diff_fb_vert':      np.full(n, avg_fb_vert - pitch_row['induced_vert']),
        'diff_fb_horz':      np.full(n, avg_fb_horz - pitch_row['horz_break']),
        'diff_spinrate':     np.full(n, avg_fb_spin - pitch_row['spin_rate']),
        'PlateLocSide':      sg.flatten(),
        'PlateLocHeight':    hg.flatten(),
    }
    df_g = pd.DataFrame(data)[pitching_features]

    pred_cols = []
    for label in labels:
        col = f'pp_{label}'
        df_g[col] = _pitching_event_models[label].predict(
            df_g[pitching_features].values,
            num_iteration=_pitching_event_models[label].best_iteration)
        pred_cols.append(col)

    df_g['lw'] = _pitching_final_model.predict(df_g[pred_cols].values)
    df_g['PitchingPlus']    = 100 + (norm_params['mu_pitching'] - df_g['lw']) / norm_params['sigma_pitching'] * 10
    df_g['WhiffPlus']       = 100 + (df_g['pp_StrikeSwinging'] - norm_params['mu_whiff_pitching'])         / norm_params['sigma_whiff_pitching']         * 10
    df_g['CalledStrikePlus']= 100 + (df_g['pp_StrikeCalled']   - norm_params['mu_called_strike_pitching']) / norm_params['sigma_called_strike_pitching'] * 10

    return side_vals, height_vals, {
        'Pitching+':     df_g['PitchingPlus'].values.reshape(grid_n, grid_n),
        'Whiff+':        df_g['WhiffPlus'].values.reshape(grid_n, grid_n),
        'CalledStrike+': df_g['CalledStrikePlus'].values.reshape(grid_n, grid_n),
    }


def compute_optimal_zone(grid, side_vals, height_vals, sd_side, sd_height, grid_n=25):
    sg, hg = np.meshgrid(side_vals, height_vals)
    cell_w = (side_vals[1] - side_vals[0])
    cell_h = (height_vals[1] - height_vals[0])
    cell_area = cell_w * cell_h
    expected_plus = np.zeros((grid_n, grid_n))
    for i, aim_h in enumerate(height_vals):
        for j, aim_s in enumerate(side_vals):
            w_side   = scipy_norm.pdf(sg, loc=aim_s, scale=sd_side)
            w_height = scipy_norm.pdf(hg, loc=aim_h, scale=sd_height)
            weights  = w_side * w_height * cell_area
            weights /= weights.sum()
            expected_plus[i, j] = (weights * grid).sum()
    best_idx = np.unravel_index(np.argmax(expected_plus), expected_plus.shape)
    best_aim_s = side_vals[best_idx[1]]
    best_aim_h = height_vals[best_idx[0]]
    best_expected = expected_plus[best_idx]
    threshold = 100.0
    w_side   = scipy_norm.pdf(sg, loc=best_aim_s, scale=sd_side)
    w_height = scipy_norm.pdf(hg, loc=best_aim_h, scale=sd_height)
    weights  = w_side * w_height * cell_area
    weights /= weights.sum()
    prob_favorable = float((weights[grid >= threshold]).sum())
    return expected_plus, best_aim_s, best_aim_h, best_expected, prob_favorable


# ------------------------------------------------------------------
# Build pitch summary from TrackMan data
# ------------------------------------------------------------------
def tm_to_pitches(tm_df, pitcher_name):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    if sub.empty: return [], 'Right'

    if 'pitcher_hand' in sub.columns:
        hand = sub['pitcher_hand'].mode()[0]
    else:
        hand = 'Right'

    pitches = []
    for pt, grp in sub.groupby('pitch_type'):
        if pt not in VALID_PITCH_TYPES: continue
        pitches.append({
            'pitch_type':   pt,
            'rel_speed':    float(grp['rel_speed'].mean()),
            'induced_vert': float(grp['induced_vert'].mean()),
            'horz_break':   float(grp['horz_break'].mean()),
            'spin_rate':    float(grp['spin_rate'].mean()),
            'extension':    float(grp['extension'].mean()),
            'rel_height':   float(grp['rel_height'].mean()),
            'rel_side':     float(grp['rel_side'].mean()),
            'count':        len(grp),
        })
    return pitches, hand


def compute_stuff_scores_from_tm(tm_df, pitcher_name, pitcher_is_righty):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    sub = sub[sub['pitch_type'].isin(VALID_PITCH_TYPES)].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(), 0, 0, 0, 0

    fb_sub = sub[sub['pitch_type'].isin(fb_types)]
    if fb_sub.empty:
        fb_sub = sub.head(1)
    avg_fb_vel  = float(fb_sub['rel_speed'].mean())
    avg_fb_vert = float(fb_sub['induced_vert'].mean())
    avg_fb_horz = float(fb_sub['horz_break'].mean())
    avg_fb_spin = float(fb_sub['spin_rate'].mean())

    feat = pd.DataFrame({
        'RelSpeed':          sub['rel_speed'],
        'InducedVertBreak':  sub['induced_vert'],
        'HorzBreak':         sub['horz_break'],
        'SpinRate':          sub['spin_rate'],
        'Extension':         sub['extension'],
        'RelHeight':         sub['rel_height'],
        'RelSide':           sub['rel_side'],
        'TaggedPitchTypeNum':sub['pitch_type'].map(pitch_type_map),
        'pitcher_is_righty': pitcher_is_righty,
        'diff_fb_vel':       avg_fb_vel  - sub['rel_speed'],
        'diff_fb_vert':      avg_fb_vert - sub['induced_vert'],
        'diff_fb_horz':      avg_fb_horz - sub['horz_break'],
        'diff_spinrate':     avg_fb_spin - sub['spin_rate'],
    })

    for label in labels:
        feat[f'sp_{label}'] = event_models[label].predict(
            feat[stuff_features].values,
            num_iteration=event_models[label].best_iteration)

    spf = [c for c in feat.columns if c.startswith('sp_')]
    feat['stuff_lw']         = final_model.predict(feat[spf].values)
    feat['StuffPlus']        = 100 + (norm_params['mu_stuff']        - feat['stuff_lw'])                 / norm_params['sigma_stuff']          * 10
    feat['WhiffPlus']        = 100 + (feat['sp_StrikeSwinging']      - norm_params['mu_whiff'])           / norm_params['sigma_whiff']          * 10
    feat['CalledStrikePlus'] = 100 + (feat['sp_StrikeCalled']        - norm_params['mu_called_strike'])   / norm_params['sigma_called_strike']  * 10
    feat['wOBACON']          = (
        (feat['sp_Single']*1 + feat['sp_Double']*2 + feat['sp_Triple']*3 + feat['sp_HomeRun']*4) /
        (feat['sp_Single'] + feat['sp_Double'] + feat['sp_Triple'] + feat['sp_HomeRun'] + 0.0001))
    feat['wOBACONPlus']      = 100 - (feat['wOBACON'] - norm_params['mu_wobacon']) / norm_params['sigma_wobacon'] * 10
    feat['pitch_type']       = sub['pitch_type'].values

    has_loc = (
        'plate_loc_side' in sub.columns and
        'plate_loc_height' in sub.columns and
        sub['plate_loc_side'].notna().any()
    )
    if has_loc:
        loc_mask = sub['plate_loc_side'].notna() & sub['plate_loc_height'].notna()
        pfeat = pd.DataFrame({
            'RelSpeed':          sub.loc[loc_mask, 'rel_speed'],
            'InducedVertBreak':  sub.loc[loc_mask, 'induced_vert'],
            'HorzBreak':         sub.loc[loc_mask, 'horz_break'],
            'SpinRate':          sub.loc[loc_mask, 'spin_rate'],
            'Extension':         sub.loc[loc_mask, 'extension'],
            'RelHeight':         sub.loc[loc_mask, 'rel_height'],
            'RelSide':           sub.loc[loc_mask, 'rel_side'],
            'TaggedPitchTypeNum':sub.loc[loc_mask, 'pitch_type'].map(pitch_type_map),
            'pitcher_is_righty': pitcher_is_righty,
            'diff_fb_vel':       avg_fb_vel  - sub.loc[loc_mask, 'rel_speed'],
            'diff_fb_vert':      avg_fb_vert - sub.loc[loc_mask, 'induced_vert'],
            'diff_fb_horz':      avg_fb_horz - sub.loc[loc_mask, 'horz_break'],
            'diff_spinrate':     avg_fb_spin - sub.loc[loc_mask, 'spin_rate'],
            'PlateLocSide':      sub.loc[loc_mask, 'plate_loc_side'],
            'PlateLocHeight':    sub.loc[loc_mask, 'plate_loc_height'],
        }).reset_index(drop=True)
        pred_cols = []
        for label in labels:
            col = f'pp_{label}'
            pfeat[col] = pitching_event_models[label].predict(
                pfeat[pitching_features].values,
                num_iteration=pitching_event_models[label].best_iteration)
            pred_cols.append(col)
        pfeat['lw'] = pitching_final_model.predict(pfeat[pred_cols].values)
        pfeat['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - pfeat['lw']) / norm_params['sigma_pitching'] * 10
        pfeat['pitch_type']   = sub.loc[loc_mask, 'pitch_type'].values
        feat.loc[loc_mask, 'PitchingPlus'] = pfeat['PitchingPlus'].values

    agg_dict = dict(
        StuffPlus        =('StuffPlus',        'mean'),
        WhiffPlus        =('WhiffPlus',        'mean'),
        CalledStrikePlus =('CalledStrikePlus', 'mean'),
        wOBACONPlus      =('wOBACONPlus',      'mean'),
        sp_StrikeSwinging=('sp_StrikeSwinging','mean'),
        sp_StrikeCalled  =('sp_StrikeCalled',  'mean'),
        sp_HomeRun       =('sp_HomeRun',       'mean'),
        count            =('StuffPlus',        'count'),
    )
    if has_loc:
        agg_dict['PitchingPlus'] = ('PitchingPlus', 'mean')
    agg = feat.groupby('pitch_type').agg(**agg_dict).reset_index()
    agg.rename(columns={'pitch_type': 'PitchType'}, inplace=True)

    return agg, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin


def tm_to_pitch_samples(tm_df, pitcher_name):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    return sub

def build_leaderboard(tm_df, pitcher_is_righty_override=None):
    has_loc = (
        'plate_loc_side' in tm_df.columns and
        'plate_loc_height' in tm_df.columns and
        tm_df['plate_loc_side'].notna().any()
    )

    rows = []
    for pitcher in tm_df['pitcher'].dropna().unique():
        sub = tm_df[tm_df['pitcher'] == pitcher]
        sub = sub[sub['pitch_type'].isin(VALID_PITCH_TYPES)].reset_index(drop=True)
        if sub.empty:
            continue

        if 'pitcher_hand' in sub.columns:
            hand = sub['pitcher_hand'].mode()[0]
        else:
            hand = 'Right'
        is_righty = 1 if hand == 'Right' else 0

        fb_sub = sub[sub['pitch_type'].isin(fb_types)]
        if fb_sub.empty:
            fb_sub = sub.head(1)
        avg_fb_vel  = float(fb_sub['rel_speed'].mean())
        avg_fb_vert = float(fb_sub['induced_vert'].mean())
        avg_fb_horz = float(fb_sub['horz_break'].mean())
        avg_fb_spin = float(fb_sub['spin_rate'].mean())

        feat = pd.DataFrame({
            'RelSpeed':          sub['rel_speed'],
            'InducedVertBreak':  sub['induced_vert'],
            'HorzBreak':         sub['horz_break'],
            'SpinRate':          sub['spin_rate'],
            'Extension':         sub['extension'],
            'RelHeight':         sub['rel_height'],
            'RelSide':           sub['rel_side'],
            'TaggedPitchTypeNum':sub['pitch_type'].map(pitch_type_map),
            'pitcher_is_righty': is_righty,
            'diff_fb_vel':       avg_fb_vel  - sub['rel_speed'],
            'diff_fb_vert':      avg_fb_vert - sub['induced_vert'],
            'diff_fb_horz':      avg_fb_horz - sub['horz_break'],
            'diff_spinrate':     avg_fb_spin - sub['spin_rate'],
        }).reset_index(drop=True)

        for label in labels:
            feat[f'sp_{label}'] = event_models[label].predict(
                feat[stuff_features].values,
                num_iteration=event_models[label].best_iteration)

        spf = [c for c in feat.columns if c.startswith('sp_')]
        feat['stuff_lw']  = final_model.predict(feat[spf].values)
        feat['StuffPlus'] = 100 + (norm_params['mu_stuff'] - feat['stuff_lw']) / norm_params['sigma_stuff'] * 10

        pitching_plus_val = np.nan
        if has_loc:
            sub_loc = sub.dropna(subset=['plate_loc_side', 'plate_loc_height']).reset_index(drop=True)
            if len(sub_loc) >= 3:
                pfeat = pd.DataFrame({
                    'RelSpeed':          sub_loc['rel_speed'],
                    'InducedVertBreak':  sub_loc['induced_vert'],
                    'HorzBreak':         sub_loc['horz_break'],
                    'SpinRate':          sub_loc['spin_rate'],
                    'Extension':         sub_loc['extension'],
                    'RelHeight':         sub_loc['rel_height'],
                    'RelSide':           sub_loc['rel_side'],
                    'TaggedPitchTypeNum':sub_loc['pitch_type'].map(pitch_type_map),
                    'pitcher_is_righty': is_righty,
                    'diff_fb_vel':       avg_fb_vel  - sub_loc['rel_speed'],
                    'diff_fb_vert':      avg_fb_vert - sub_loc['induced_vert'],
                    'diff_fb_horz':      avg_fb_horz - sub_loc['horz_break'],
                    'diff_spinrate':     avg_fb_spin - sub_loc['spin_rate'],
                    'PlateLocSide':      sub_loc['plate_loc_side'],
                    'PlateLocHeight':    sub_loc['plate_loc_height'],
                }).reset_index(drop=True)

                pred_cols = []
                for label in labels:
                    col = f'pp_{label}'
                    pfeat[col] = pitching_event_models[label].predict(
                        pfeat[pitching_features].values,
                        num_iteration=pitching_event_models[label].best_iteration)
                    pred_cols.append(col)

                pfeat['lw'] = pitching_final_model.predict(pfeat[pred_cols].values)
                pfeat['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - pfeat['lw']) / norm_params['sigma_pitching'] * 10
                pitching_plus_val = float(pfeat['PitchingPlus'].mean())

        total = len(sub)
        row = {
            'Pitcher':    pitcher,
            'Hand':       'R' if is_righty else 'L',
            'Pitches':    total,
            'Max Velo':   fmt(float(sub['rel_speed'].max())),
            'Stuff+':     fmt(float(feat['StuffPlus'].mean())),
        }
        if has_loc:
            row['Pitching+'] = fmt(pitching_plus_val) if not np.isnan(pitching_plus_val) else np.nan
        rows.append(row)

    sort_col = 'Pitching+' if has_loc and any(r.get('Pitching+') is not None for r in rows) else 'Stuff+'
    lb = pd.DataFrame(rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
    lb.insert(0, 'Rank', range(1, len(lb)+1))
    return lb, sort_col


# ------------------------------------------------------------------
# 2025 Season summary loader (cached)
# ------------------------------------------------------------------
@st.cache_data
def load_season_summary():
    try:
        return pd.read_csv("pitcher_overall_summary_2025.csv")
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚾ Stuff+ Builder")
    input_mode = st.radio(
        "Mode",
        ["Arsenal Builder", "TrackMan Analyzer", "2025 Season"],
        index=["Arsenal Builder", "TrackMan Analyzer", "2025 Season"].index(
            st.session_state.input_mode if st.session_state.input_mode in ["Arsenal Builder", "TrackMan Analyzer", "2025 Season"] else "Arsenal Builder"
        ),
        help="Arsenal Builder: manually design a pitch arsenal.  TrackMan Analyzer: upload a game CSV to analyze real pitchers.  2025 Season: full season leaderboard."
    )
    st.session_state.input_mode = input_mode
    st.divider()

    if input_mode == "2025 Season":
        st.caption("Full 2025 season pitcher leaderboard.")

    elif input_mode == "Arsenal Builder":
        pitcher_name      = st.text_input("Name", value="My Arsenal", label_visibility="collapsed", placeholder="Pitcher name")
        pitcher_hand      = st.radio("Throws", ["Right", "Left"], horizontal=True)
        pitcher_is_righty = 1 if pitcher_hand == "Right" else 0
        if st.session_state.pitches:
            st.caption(f"{len(st.session_state.pitches)} pitch(es) in arsenal")

    elif input_mode == "TrackMan Analyzer":
        st.markdown("#### Upload TrackMan CSV")
        uploaded_file = st.file_uploader("", type=["csv"], key="tm_upload", label_visibility="collapsed")
        if uploaded_file:
            with st.spinner("Reading file..."):
                tm_data, err = parse_trackman_csv(uploaded_file)
            if err:
                st.error(f"Could not parse file: {err}")
                st.session_state.tm_data = None
            else:
                st.session_state.tm_data = tm_data
                pitchers = sorted(tm_data["pitcher"].unique())
                st.success(f"{len(tm_data):,} pitches · {len(pitchers)} pitcher(s) found")

        if st.session_state.tm_data is not None:
            tm_data = st.session_state.tm_data
            pitchers = sorted(tm_data["pitcher"].unique())
            selected_pitcher = st.selectbox("Select Pitcher", pitchers)
        else:
            selected_pitcher = None

    st.divider()
    st.caption("100 = college baseball average  |  110+ = elite  |  <90 = below avg")
    st.caption("Horizontal break: + = arm side  |  − = glove side")
    st.caption("All charts shown from pitcher's view")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
st.title("Stuff+ Arsenal Builder")

# =========================================================
# 2025 SEASON MODE
# =========================================================
if st.session_state.input_mode == "2025 Season":
    season_df = load_season_summary()
    if season_df is None:
        st.warning("⚠️  `pitcher_overall_summary_2025.csv` not found. Run the training script first to generate it.")
    else:
        has_pitching = (
            "Overall_PitchingPlus" in season_df.columns and
            season_df["Overall_PitchingPlus"].notna().any()
        )

        disp_cols = ["Pitcher", "PitcherTeam", "Total_Pitches", "Overall_StuffPlus"]
        if has_pitching:
            disp_cols.append("Overall_PitchingPlus")

        disp_s = season_df[disp_cols].copy()
        rename_map = {
            "Pitcher":           "Pitcher",
            "PitcherTeam":       "Team",
            "Total_Pitches":     "Pitches",
            "Overall_StuffPlus": "Stuff+",
        }
        if has_pitching:
            rename_map["Overall_PitchingPlus"] = "Pitching+"
        disp_s = disp_s.rename(columns=rename_map)

        if "PitcherThrows" in season_df.columns:
            disp_s.insert(1, "Hand", season_df["PitcherThrows"].map(
                lambda x: "R" if str(x).strip() in ["Right", "R", "RHP"] else "L"
            ))

        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
        with fc1:
            sort_options = ["Stuff+"] + (["Pitching+"] if has_pitching else [])
            sort_by = st.selectbox("Sort by", sort_options, key="season_sort")
        with fc2:
            team_options = ["All Teams"] + sorted(disp_s["Team"].dropna().unique().tolist())
            selected_team = st.selectbox("Team", team_options, key="season_team")
        with fc3:
            name_search = st.text_input("Search pitcher name", value="", placeholder="e.g. Smith", key="season_name")
        with fc4:
            min_pitches = st.number_input("Min pitches", min_value=1, value=50, step=10, key="season_min_p")

        disp_s = disp_s[disp_s["Pitches"] >= min_pitches]
        if selected_team != "All Teams":
            disp_s = disp_s[disp_s["Team"] == selected_team]
        if name_search.strip():
            disp_s = disp_s[disp_s["Pitcher"].str.contains(name_search.strip(), case=False, na=False)]

        disp_s = disp_s.sort_values(sort_by, ascending=False).reset_index(drop=True)
        disp_s.insert(0, "Rank", range(1, len(disp_s) + 1))

        for col in ["Stuff+", "Pitching+"]:
            if col in disp_s.columns:
                disp_s[col] = disp_s[col].round(1)

        def color_grade_s(val):
            try:
                v = float(val)
                if v >= 110: return "background-color: #1B5E20; color: white; font-weight: bold"
                if v >= 105: return "background-color: #388E3C; color: white"
                if v >= 100: return "background-color: #81C784; color: black"
                if v >= 95:  return "background-color: #FDD835; color: black"
                if v >= 90:  return "background-color: #FFB74D; color: black"
                return              "background-color: #E53935; color: white"
            except:
                return ""

        grade_cols_s = [c for c in ["Stuff+", "Pitching+"] if c in disp_s.columns]

        # Weighted averages across the filtered dataset
        if not disp_s.empty:
            w = disp_s["Pitches"].values
            wa_stuff = float(np.average(disp_s["Stuff+"].values, weights=w))
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Avg Stuff+ (weighted)", f"{wa_stuff:.1f}", f"{wa_stuff-100:+.1f}")
            if "Pitching+" in disp_s.columns and disp_s["Pitching+"].notna().any():
                pp_mask = disp_s["Pitching+"].notna()
                wa_pitching = float(np.average(disp_s.loc[pp_mask, "Pitching+"].values, weights=disp_s.loc[pp_mask, "Pitches"].values))
                sm2.metric("Avg Pitching+ (weighted)", f"{wa_pitching:.1f}", f"{wa_pitching-100:+.1f}")
            sm3.metric("Total Pitches in View", f"{int(disp_s['Pitches'].sum()):,}")

        st.markdown(f"**{len(disp_s)} pitchers** · min {min_pitches} pitches · sorted by {sort_by}")
        st.dataframe(
            disp_s.style.applymap(color_grade_s, subset=grade_cols_s),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("100 = college average  ·  110+ = elite  ·  <90 = below average")

        csv_bytes = disp_s.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Download CSV",
            data=csv_bytes,
            file_name="pitcher_summary_2025_export.csv",
            mime="text/csv",
        )
    st.stop()

# =========================================================
# MANUAL MODE
# =========================================================
if st.session_state.input_mode == "Arsenal Builder":

    pitch_type = st.selectbox("Pitch Type",
        ["Fastball", "Sinker", "Slider", "Curveball", "ChangeUp", "Cutter", "Splitter"])

    defaults   = PITCH_DEFAULTS[pitch_type]
    default_hb = defaults["hb"] if pitcher_is_righty else -defaults["hb"]

    col1, col2, col3, col4 = st.columns(4)
    with col1: rel_speed    = st.slider("Velocity (mph)",    65.0, 105.0, float(defaults["velo"]), 0.5)
    with col2: induced_vert = st.slider("Induced Vert (in)", -20.0, 25.0, float(defaults["ivb"]),  0.5)
    with col3: horz_break   = st.slider("Horz Break (in)",  -25.0, 25.0, float(default_hb),        0.5)
    with col4: spin_rate    = st.slider("Spin Rate (rpm)",   1000,  3500, int(defaults["spin"]),    50)

    with st.expander("Release Settings"):
        c1, c2, c3 = st.columns(3)
        with c1: extension  = st.slider("Extension (ft)",      4.0, 7.5, 6.0, 0.1)
        with c2: rel_height = st.slider("Release Height (ft)", 4.0, 7.0, 5.5, 0.1)
        with c3:
            default_rel_side = 2.0 if pitcher_is_righty else -2.0
            rel_side = st.slider("Release Side (ft)", -3.0, 3.0, default_rel_side, 0.1)

    if st.button("➕ Add Pitch", use_container_width=True, type="primary"):
        st.session_state.pitches.append({
            'pitch_type':   pitch_type,
            'rel_speed':    rel_speed,
            'induced_vert': induced_vert,
            'horz_break':   horz_break,
            'spin_rate':    spin_rate,
            'extension':    extension,
            'rel_height':   rel_height,
            'rel_side':     rel_side,
        })
        st.toast(f"✅ {pitch_type} added to arsenal")

    pitches_to_show   = st.session_state.pitches
    pitcher_is_righty_display = pitcher_is_righty

# =========================================================
# TRACKMAN MODE
# =========================================================
else:
    if st.session_state.tm_data is None or selected_pitcher is None:
        st.info("👈  Upload a TrackMan CSV using the sidebar to get started.")
        st.stop()

    tm_data = st.session_state.tm_data
    pitches_list, pitcher_hand = tm_to_pitches(tm_data, selected_pitcher)
    pitcher_is_righty_display  = 1 if pitcher_hand == 'Right' else 0

    if not pitches_list:
        st.warning(f"No valid pitches found for {selected_pitcher}")
        st.stop()

    pitcher_name = selected_pitcher
    pitches_to_show = pitches_list

    sub = tm_data[tm_data['pitcher'] == selected_pitcher]
    with st.expander(f"📋  Raw Pitch Data — {selected_pitcher} ({len(sub)} pitches)", expanded=False):
        pt_counts = sub['pitch_type'].value_counts().reset_index()
        pt_counts.columns = ['Pitch Type', 'Count']
        pt_counts['%'] = (pt_counts['Count'] / pt_counts['Count'].sum() * 100).round(1)
        st.dataframe(pt_counts, use_container_width=True, hide_index=True)

        has_loc = ('plate_loc_side' in sub.columns) and sub['plate_loc_side'].notna().any()
        if has_loc:
            st.markdown("**Pitch Locations**")
            pitch_colors = {
                'Fastball': '#E53935', 'Sinker': '#FF7043', 'Slider': '#8E24AA',
                'Curveball': '#1E88E5', 'ChangeUp': '#43A047', 'Cutter': '#FFB300', 'Splitter': '#00ACC1'
            }
            fig_raw = go.Figure()
            for pt in sub['pitch_type'].dropna().unique():
                grp = sub[sub['pitch_type'] == pt].dropna(subset=['plate_loc_side', 'plate_loc_height'])
                if grp.empty: continue
                fig_raw.add_trace(go.Scatter(
                    x=grp['plate_loc_side'], y=grp['plate_loc_height'],
                    mode='markers', name=pt,
                    marker=dict(size=6, color=pitch_colors.get(pt, 'white'), opacity=0.7),
                    hovertemplate=f"{pt}<br>Side: %{{x:.2f}}<br>Height: %{{y:.2f}}<extra></extra>"
                ))
            fig_raw.add_trace(go.Scatter(
                x=[-0.708, 0.708, 0.708, -0.708, -0.708],
                y=[1.5, 1.5, 3.5, 3.5, 1.5],
                mode='lines', line=dict(color='white', width=2), hoverinfo='skip', showlegend=False
            ))
            fig_raw.update_layout(
                xaxis=dict(title="Plate Side (ft)", range=[-3.5, 3.5]),
                yaxis=dict(title="Height (ft)", range=[-0.5, 6.0]),
                height=450, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='white'), legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig_raw, use_container_width=True)


# ------------------------------------------------------------------
# Leaderboard (TrackMan mode only)
# ------------------------------------------------------------------
if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
    with st.expander("🏆  Pitcher Leaderboard", expanded=True):
        with st.spinner("Scoring all pitchers..."):
            lb, sort_col = build_leaderboard(st.session_state.tm_data)
        if lb.empty:
            st.info("No scorable pitchers found.")
        else:
            def color_grade(val):
                try:
                    v = float(val)
                    if v >= 110: return 'background-color: #1B5E20; color: white; font-weight: bold'
                    if v >= 105: return 'background-color: #388E3C; color: white'
                    if v >= 100: return 'background-color: #81C784; color: black'
                    if v >= 95:  return 'background-color: #FDD835; color: black'
                    if v >= 90:  return 'background-color: #FFB74D; color: black'
                    return              'background-color: #E53935; color: white'
                except: return ''
            for col in ['Max Velo', 'Stuff+', 'Pitching+']:
                if col in lb.columns:
                    lb[col] = lb[col].apply(lambda x: str(fmt(x)).rstrip('0').rstrip('.') if pd.notna(x) else x)
            grade_cols = [c for c in ['Stuff+', 'Pitching+'] if c in lb.columns]
            st.dataframe(
                lb.style.applymap(color_grade, subset=grade_cols),
                use_container_width=True, hide_index=True)
            has_pitching = 'Pitching+' in lb.columns
            st.caption(f"Ranked by {sort_col}  ·  100 = college baseball average" + ("  ·  Upload file with PlateLocSide/Height for Pitching+" if not has_pitching else ""))

# ------------------------------------------------------------------
# Arsenal display (shared between modes)
# ------------------------------------------------------------------
if pitches_to_show:
    st.divider()
    st.subheader(f"{pitcher_name}  —  {'RHP' if pitcher_is_righty_display else 'LHP'}")

    if st.session_state.input_mode == "TrackMan Analyzer":
        tm_scores, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin = compute_stuff_scores_from_tm(
            st.session_state.tm_data, pitcher_name, pitcher_is_righty_display)
        chars = pd.DataFrame([{
            'PitchType':        p['pitch_type'],
            'RelSpeed':         p['rel_speed'],
            'InducedVertBreak': p['induced_vert'],
            'HorzBreak':        p['horz_break'],
            'SpinRate':         p['spin_rate'],
        } for p in pitches_to_show])
        df = pd.merge(chars, tm_scores, on='PitchType', how='left')
        ordered_types = [p['pitch_type'] for p in pitches_to_show]
        df = df.set_index('PitchType').loc[ordered_types].reset_index()
    else:
        df, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin = compute_stuff_scores(
            pitches_to_show, pitcher_is_righty_display)

    if st.session_state.input_mode == "TrackMan Analyzer" and 'count' in df.columns:
        w = df['count'].fillna(1).values
        def wavg(col): return float(np.average(df[col].fillna(0), weights=w))
        stuff_val = wavg('StuffPlus')
        m1, m2, m3 = st.columns(3)
        m1.metric("Stuff+", f"{stuff_val:.0f}", f"{stuff_val-100:+.0f}")
        if 'PitchingPlus' in df.columns and df['PitchingPlus'].notna().any():
            pp_val = wavg('PitchingPlus')
            m2.metric("Pitching+", f"{pp_val:.0f}", f"{pp_val-100:+.0f}")
        else:
            m2.metric("Pitching+", "N/A", help="Requires plate location data")
        max_velo = float(st.session_state.tm_data[
            st.session_state.tm_data['pitcher'] == pitcher_name
        ]['rel_speed'].max())
        m3.metric("Max Velo", f"{max_velo:.1f} mph")
    else:
        def wavg(col): return float(df[col].mean())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stuff+",        f"{wavg('StuffPlus'):.0f}",        f"{wavg('StuffPlus')-100:+.0f}")
        m2.metric("Whiff+",        f"{wavg('WhiffPlus'):.0f}",        f"{wavg('WhiffPlus')-100:+.0f}")
        m3.metric("CalledStrike+", f"{wavg('CalledStrikePlus'):.0f}", f"{wavg('CalledStrikePlus')-100:+.0f}")
        m4.metric("wOBACON+",      f"{wavg('wOBACONPlus'):.0f}",      f"{wavg('wOBACONPlus')-100:+.0f}")

    pitch_counts_fb = {}
    for p in pitches_to_show:
        if p['pitch_type'] in fb_types:
            pitch_counts_fb[p['pitch_type']] = pitch_counts_fb.get(p['pitch_type'], 0) + 1

    tab_options = ["📊 Grades", "🎯 Movement", "🟩 Zone"]
    if st.session_state.input_mode == "TrackMan Analyzer":
        tab_options.append("⚾ Pitch Plot")

    if st.session_state.active_tab >= len(tab_options):
        st.session_state.active_tab = 0

    selected_tab = st.radio(
        "View", tab_options,
        index=st.session_state.active_tab,
        horizontal=True, key="tab_radio", label_visibility="collapsed"
    )
    st.session_state.active_tab = tab_options.index(selected_tab)

    # ---- Grades ----
    if selected_tab == "📊 Grades":
        has_pitching_col = (
            st.session_state.input_mode == "TrackMan Analyzer" and
            'PitchingPlus' in df.columns and
            df['PitchingPlus'].notna().any()
        )
        disp_cols = ['PitchType', 'RelSpeed', 'InducedVertBreak', 'HorzBreak',
                     'StuffPlus', 'WhiffPlus', 'CalledStrikePlus', 'wOBACONPlus']
        if has_pitching_col:
            disp_cols.insert(disp_cols.index('WhiffPlus'), 'PitchingPlus')
        disp = df[disp_cols].copy()
        base_cols = ['Pitch', 'Velo', 'IVB', 'HB', 'Stuff+', 'Whiff+', 'CalledStrike+', 'wOBACON+']
        if has_pitching_col:
            base_cols.insert(base_cols.index('Whiff+'), 'Pitching+')
        disp.columns = base_cols

        if st.session_state.input_mode == "TrackMan Analyzer":
            disp['N'] = [p.get('count', 0) for p in pitches_to_show]
            disp = disp.sort_values('N', ascending=False).reset_index(drop=True)
            disp.insert(1, 'N', disp.pop('N'))

        for col in disp.columns:
            if col not in ('Pitch', 'N'): disp[col] = disp[col].apply(fmt).apply(str).str.replace(r'\.0$', '', regex=True)
        disp.insert(0, '#', range(1, len(disp)+1))

        def color_grade(val):
            try:
                v = float(val)
                if v >= 110: return 'background-color: #1B5E20; color: white; font-weight: bold'
                if v >= 105: return 'background-color: #388E3C; color: white'
                if v >= 100: return 'background-color: #81C784; color: black'
                if v >= 95:  return 'background-color: #FDD835; color: black'
                if v >= 90:  return 'background-color: #FFB74D; color: black'
                return              'background-color: #E53935; color: white'
            except: return ''

        grade_cols = [c for c in ['Stuff+', 'Pitching+', 'Whiff+', 'CalledStrike+', 'wOBACON+'] if c in disp.columns]
        st.dataframe(
            disp.style.applymap(color_grade, subset=grade_cols),
            use_container_width=True, hide_index=True)
        st.caption("100 = college average  ·  Higher is better  ·  Color scale: green = elite, red = below average")

        for idx, p in enumerate(pitches_to_show):
            label = f"#{idx+1} {p['pitch_type']} — {p['rel_speed']:.1f} mph"
            if 'count' in p: label += f" ({p['count']} pitches)"
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Pitch Characteristics**")
                    st.write(f"Velo: {p['rel_speed']:.1f} mph")
                    st.write(f"IVB: {p['induced_vert']:+.1f}\"")
                    st.write(f"HB: {p['horz_break']:+.1f}\"")
                    st.write(f"Spin: {p['spin_rate']:.0f} rpm")
                with c2:
                    st.markdown("**Grades**")
                    st.write(f"Stuff+: {disp.iloc[idx]['Stuff+']}")
                    if 'Pitching+' in disp.columns:
                        st.write(f"Pitching+: {disp.iloc[idx]['Pitching+']}")
                    st.write(f"Whiff+: {disp.iloc[idx]['Whiff+']}")
                    st.write(f"CalledStrike+: {disp.iloc[idx]['CalledStrike+']}")
                    st.write(f"wOBACON+: {disp.iloc[idx]['wOBACON+']}")
                with c3:
                    st.markdown("**Estimated Probabilities**")
                    st.write(f"Whiff: {df.iloc[idx]['sp_StrikeSwinging']*100:.1f}%")
                    st.write(f"Called Strike: {df.iloc[idx]['sp_StrikeCalled']*100:.1f}%")
                    st.write(f"Home Run: {df.iloc[idx]['sp_HomeRun']*100:.1f}%")
                if st.session_state.input_mode == "Arsenal Builder":
                    if st.button(f"Remove #{idx+1}", key=f"del_{idx}"):
                        st.session_state.pitches.pop(idx)
                        st.rerun()

    # ---- Movement ----
    elif selected_tab == "🎯 Movement":
        st.caption("Movement plot — pitcher's view  ·  Dot color shows Stuff+ grade")
        fig = go.Figure()

        if st.session_state.input_mode == "TrackMan Analyzer":
            show_scatter = st.checkbox("Show individual pitch locations", value=False, help="Overlay every pitch from the game as a faint dot")
        else:
            show_scatter = False

        pitch_colors_scatter = {
            'Fastball': 'rgba(229,57,53,0.15)', 'Sinker': 'rgba(255,112,67,0.15)',
            'Slider': 'rgba(142,36,170,0.15)', 'Curveball': 'rgba(30,136,229,0.15)',
            'ChangeUp': 'rgba(67,160,71,0.15)', 'Cutter': 'rgba(255,179,0,0.15)',
            'Splitter': 'rgba(0,172,193,0.15)'
        }

        if show_scatter and st.session_state.tm_data is not None:
            sub = tm_data[tm_data['pitcher'] == selected_pitcher]
            for pt in sub['pitch_type'].dropna().unique():
                grp = sub[sub['pitch_type'] == pt].dropna(subset=['horz_break', 'induced_vert'])
                if grp.empty: continue
                fig.add_trace(go.Scatter(
                    x=grp['horz_break'], y=grp['induced_vert'],
                    mode='markers', name=f"{pt} (raw)",
                    marker=dict(size=5, color=pitch_colors_scatter.get(pt, 'rgba(255,255,255,0.1)')),
                    hoverinfo='skip', showlegend=False
                ))

        for idx, p in enumerate(pitches_to_show):
            sv    = df.iloc[idx]['StuffPlus']
            color = '#1B5E20' if sv >= 105 else '#81C784' if sv >= 95 else '#FFB74D'
            label = p['pitch_type']
            if 'count' in p: label += f" ({p['count']})"
            fig.add_trace(go.Scatter(
                x=[p['horz_break']], y=[p['induced_vert']],
                mode='markers+text',
                text=[f"{label}<br>{sv:.0f}"],
                textposition="top center",
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                hovertemplate=f"<b>{p['pitch_type']}</b><br>{p['rel_speed']:.1f} mph<br>Stuff+: {sv:.0f}<extra></extra>"
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig.update_layout(
            xaxis_title="Horizontal Break — inches (pitcher's view)",
            yaxis_title="Induced Vertical Break (inches)",
            xaxis=dict(range=[-25, 25]),
            yaxis=dict(range=[-25, 25]),
            height=580, showlegend=False,
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Zone Heatmap ----
    elif selected_tab == "🟩 Zone":
        st.caption("Where each pitch plays best in the zone — pitcher's view  ·  Green = better location, Red = worse")

        pitch_labels = [f"#{i+1} {p['pitch_type']} ({p['rel_speed']:.1f} mph)"
                        + (f" [{p['count']}]" if 'count' in p else "")
                        for i, p in enumerate(pitches_to_show)]

        zc1, zc2 = st.columns([2, 1])
        with zc1:
            selected_label = st.selectbox("Pitch", pitch_labels, key="zone_pitch_select")
        with zc2:
            zone_metric = st.selectbox("Metric", ["Pitching+", "Whiff+", "CalledStrike+"], key="zone_metric_select")

        sel_idx        = pitch_labels.index(selected_label)
        selected_pitch = pitches_to_show[sel_idx]

        with st.spinner("Computing..."):
            side_vals, height_vals, grids = compute_zone_heatmap(
                pitch_row=selected_pitch,
                avg_fb_vel=avg_fb_vel, avg_fb_vert=avg_fb_vert,
                avg_fb_horz=avg_fb_horz, avg_fb_spin=avg_fb_spin,
                pitcher_is_righty=pitcher_is_righty_display,
                _pitching_event_models=pitching_event_models,
                _pitching_final_model=pitching_final_model,
                norm_params=norm_params, pitching_features=pitching_features,
                pitch_type_map=pitch_type_map, labels=labels, grid_n=25
            )

        grid     = grids[zone_metric]
        peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
        peak_s   = side_vals[peak_idx[1]]
        peak_h   = height_vals[peak_idx[0]]
        peak_v   = grid[peak_idx]

        hand_str  = "Right" if pitcher_is_righty_display else "Left"
        pitch_str = selected_pitch['pitch_type']
        sd_row    = location_sd_params[
            (location_sd_params['PitcherThrows'] == hand_str) &
            (location_sd_params['TaggedPitchType'] == pitch_str)
        ]
        if len(sd_row) == 0:
            sd_row = location_sd_params[location_sd_params['PitcherThrows'] == hand_str]
        sd_side   = float(sd_row['avg_SD_PlateLocSide'].mean())   if len(sd_row) > 0 else 0.25
        sd_height = float(sd_row['avg_SD_PlateLocHeight'].mean()) if len(sd_row) > 0 else 0.25

        if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
            sub_pt = tm_data[
                (tm_data['pitcher'] == selected_pitcher) &
                (tm_data['pitch_type'] == pitch_str)
            ].dropna(subset=['plate_loc_side', 'plate_loc_height'])
            if len(sub_pt) >= 5:
                sd_side_actual   = float(sub_pt['plate_loc_side'].std())
                sd_height_actual = float(sub_pt['plate_loc_height'].std())
                use_actual = st.checkbox(
                    f"Use actual command SDs from this game "
                    f"(±{sd_side_actual:.3f} / ±{sd_height_actual:.3f} ft)", value=False,
                    key="use_actual_sd"
                )
                if use_actual:
                    sd_side   = sd_side_actual
                    sd_height = sd_height_actual

        expected_grid, best_aim_s, best_aim_h, best_expected, prob_favorable = compute_optimal_zone(
            grid, side_vals, height_vals, sd_side, sd_height, grid_n=25
        )

        st.caption(f"Command scatter used: ±{sd_side:.2f} ft horizontal  ·  ±{sd_height:.2f} ft vertical  ({hand_str}HP {pitch_str})")

        view_mode = st.radio(
            "Map view",
            ["Raw heatmap", "Expected value (accounts for command scatter)"],
            horizontal=True, key="zone_view_mode"
        )

        display_grid   = grid if view_mode == "Raw heatmap" else expected_grid
        colorbar_title = zone_metric if view_mode == "Raw heatmap" else f"Expected {zone_metric}"

        fig_z = go.Figure()
        fig_z.add_trace(go.Contour(
            x=side_vals, y=height_vals, z=display_grid,
            colorscale='RdYlGn',
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            colorbar=dict(title=colorbar_title, thickness=15),
            hovertemplate=f"Side: %{{x:.2f}} ft<br>Height: %{{y:.2f}} ft<br>{colorbar_title}: %{{z:.1f}}<extra></extra>",
            line_smoothing=0.85,
        ))
        fig_z.add_trace(go.Scatter(
            x=[-0.708, 0.708, 0.708, -0.708, -0.708],
            y=[1.5, 1.5, 3.5, 3.5, 1.5],
            mode='lines', line=dict(color='white', width=2.5), hoverinfo='skip', showlegend=False
        ))
        fig_z.add_trace(go.Scatter(
            x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708],
            y=[0.0, 0.0, 0.15, 0.22, 0.15, 0.0],
            mode='lines', line=dict(color='lightgray', width=1.5), hoverinfo='skip', showlegend=False
        ))

        if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
            sub_pt = tm_data[
                (tm_data['pitcher'] == selected_pitcher) &
                (tm_data['pitch_type'] == pitch_str)
            ].dropna(subset=['plate_loc_side', 'plate_loc_height'])
            if not sub_pt.empty:
                fig_z.add_trace(go.Scatter(
                    x=sub_pt['plate_loc_side'], y=sub_pt['plate_loc_height'],
                    mode='markers', name='Actual locations',
                    marker=dict(size=5, color='white', opacity=0.4,
                                line=dict(color='gray', width=0.5)),
                    hovertemplate="Actual<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>"
                ))

        fig_z.add_shape(type='line',
            x0=best_aim_s, x1=best_aim_s, y0=best_aim_h-0.35, y1=best_aim_h+0.35,
            line=dict(color='cyan', width=2))
        fig_z.add_shape(type='line',
            x0=best_aim_s-0.25, x1=best_aim_s+0.25, y0=best_aim_h, y1=best_aim_h,
            line=dict(color='cyan', width=2))

        theta = np.linspace(0, 2*np.pi, 100)
        fig_z.add_trace(go.Scatter(
            x=best_aim_s + sd_side   * np.cos(theta),
            y=best_aim_h + sd_height * np.sin(theta),
            mode='lines', line=dict(color='cyan', width=1.5, dash='dot'),
            hoverinfo='skip', name='1 SD ellipse'
        ))
        fig_z.add_trace(go.Scatter(
            x=best_aim_s + 2*sd_side   * np.cos(theta),
            y=best_aim_h + 2*sd_height * np.sin(theta),
            mode='lines', line=dict(color='cyan', width=1, dash='dash'),
            hoverinfo='skip', name='2 SD ellipse'
        ))

        if view_mode == "Raw heatmap":
            fig_z.add_trace(go.Scatter(
                x=[peak_s], y=[peak_h],
                mode='markers+text',
                marker=dict(symbol='star', size=18, color='gold', line=dict(color='black', width=1)),
                text=[f"  {peak_v:.1f}"], textfont=dict(color='white', size=11),
                textposition='middle right',
                hovertemplate=f"Peak {zone_metric}: {peak_v:.1f}<extra></extra>",
                name='Peak location'
            ))

        fig_z.add_trace(go.Scatter(
            x=[best_aim_s], y=[best_aim_h],
            mode='markers',
            marker=dict(symbol='circle', size=10, color='cyan', line=dict(color='black', width=1.5)),
            hovertemplate=f"Optimal aim<br>Expected {zone_metric}: {best_expected:.1f}<br>P(Favorable): {prob_favorable*100:.1f}%<extra></extra>",
            name='Optimal aim'
        ))

        fig_z.update_layout(
            xaxis=dict(title="Horizontal (ft, pitcher's view)", range=[-3.5, 3.5], zeroline=False, tickformat='.1f'),
            yaxis=dict(title="Height (ft)", range=[-0.7, 6.0], zeroline=False, scaleanchor='x', scaleratio=1),
            height=600, showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(color='white', size=10)),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='white'),
            title=dict(text=f"{selected_pitch['pitch_type']} — {colorbar_title} by Location", font=dict(size=14))
        )
        st.plotly_chart(fig_z, use_container_width=True)

        with st.expander("ℹ️  How to read this chart"):
            st.markdown(f"""
**Crosshair (cyan)** — optimal aim point that maximizes *expected* {zone_metric} 
accounting for command scatter from a {hand_str}HP {pitch_str}.

**Dotted ellipse** — 1 SD spread (~68% of pitches land inside).  
**Dashed ellipse** — 2 SD spread (~95% of pitches land inside).  
**White dots** — actual pitch locations from this TrackMan game (if available).

**Expected {zone_metric}: {best_expected:.1f}** — average grade given typical miss 
of ±{sd_side:.2f} ft horizontal / ±{sd_height:.2f} ft vertical.

**P(Favorable): {prob_favorable*100:.1f}%** — probability of landing in a zone ≥ 100.
""")

    # ---- Pitch Plot ----
    elif selected_tab == "⚾ Pitch Plot":
        if st.session_state.tm_data is None:
            st.info("No TrackMan data loaded.")
        else:
            tm_sub = st.session_state.tm_data[
                st.session_state.tm_data['pitcher'] == pitcher_name
            ].copy()
            tm_sub = tm_sub[tm_sub['pitch_type'].isin(VALID_PITCH_TYPES)]
            has_loc_plot = (
                'plate_loc_side' in tm_sub.columns and
                'plate_loc_height' in tm_sub.columns and
                tm_sub['plate_loc_side'].notna().any()
            )
            if not has_loc_plot:
                st.warning("No plate location data found in this TrackMan file.")
            else:
                fb_sub_p = tm_sub[tm_sub['pitch_type'].isin(fb_types)]
                if fb_sub_p.empty: fb_sub_p = tm_sub.head(1)
                pfb_vel  = float(fb_sub_p['rel_speed'].mean())
                pfb_vert = float(fb_sub_p['induced_vert'].mean())
                pfb_horz = float(fb_sub_p['horz_break'].mean())
                pfb_spin = float(fb_sub_p['spin_rate'].mean())

                pfeat = pd.DataFrame({
                    'RelSpeed':          tm_sub['rel_speed'],
                    'InducedVertBreak':  tm_sub['induced_vert'],
                    'HorzBreak':         tm_sub['horz_break'],
                    'SpinRate':          tm_sub['spin_rate'],
                    'Extension':         tm_sub['extension'],
                    'RelHeight':         tm_sub['rel_height'],
                    'RelSide':           tm_sub['rel_side'],
                    'TaggedPitchTypeNum':tm_sub['pitch_type'].map(pitch_type_map),
                    'pitcher_is_righty': pitcher_is_righty_display,
                    'diff_fb_vel':       pfb_vel  - tm_sub['rel_speed'],
                    'diff_fb_vert':      pfb_vert - tm_sub['induced_vert'],
                    'diff_fb_horz':      pfb_horz - tm_sub['horz_break'],
                    'diff_spinrate':     pfb_spin - tm_sub['spin_rate'],
                }).reset_index(drop=True)

                for label in labels:
                    pfeat[f'sp_{label}'] = event_models[label].predict(
                        pfeat[stuff_features].values,
                        num_iteration=event_models[label].best_iteration)
                spf = [c for c in pfeat.columns if c.startswith('sp_')]
                pfeat['stuff_lw']  = final_model.predict(pfeat[spf].values)
                pfeat['StuffPlus'] = 100 + (norm_params['mu_stuff'] - pfeat['stuff_lw']) / norm_params['sigma_stuff'] * 10

                loc_mask = tm_sub['plate_loc_side'].notna() & tm_sub['plate_loc_height'].notna()
                pfeat['PitchingPlus'] = np.nan
                if loc_mask.any():
                    pp_feat = pd.DataFrame({
                        'RelSpeed':          tm_sub.loc[loc_mask, 'rel_speed'],
                        'InducedVertBreak':  tm_sub.loc[loc_mask, 'induced_vert'],
                        'HorzBreak':         tm_sub.loc[loc_mask, 'horz_break'],
                        'SpinRate':          tm_sub.loc[loc_mask, 'spin_rate'],
                        'Extension':         tm_sub.loc[loc_mask, 'extension'],
                        'RelHeight':         tm_sub.loc[loc_mask, 'rel_height'],
                        'RelSide':           tm_sub.loc[loc_mask, 'rel_side'],
                        'TaggedPitchTypeNum':tm_sub.loc[loc_mask, 'pitch_type'].map(pitch_type_map),
                        'pitcher_is_righty': pitcher_is_righty_display,
                        'diff_fb_vel':       pfb_vel  - tm_sub.loc[loc_mask, 'rel_speed'],
                        'diff_fb_vert':      pfb_vert - tm_sub.loc[loc_mask, 'induced_vert'],
                        'diff_fb_horz':      pfb_horz - tm_sub.loc[loc_mask, 'horz_break'],
                        'diff_spinrate':     pfb_spin - tm_sub.loc[loc_mask, 'spin_rate'],
                        'PlateLocSide':      tm_sub.loc[loc_mask, 'plate_loc_side'],
                        'PlateLocHeight':    tm_sub.loc[loc_mask, 'plate_loc_height'],
                    }).reset_index(drop=True)
                    pp_pred_cols = []
                    for label in labels:
                        col = f'pp_{label}'
                        pp_feat[col] = pitching_event_models[label].predict(
                            pp_feat[pitching_features].values,
                            num_iteration=pitching_event_models[label].best_iteration)
                        pp_pred_cols.append(col)
                    pp_feat['lw'] = pitching_final_model.predict(pp_feat[pp_pred_cols].values)
                    pp_feat['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - pp_feat['lw']) / norm_params['sigma_pitching'] * 10
                    pfeat.loc[loc_mask.values[:len(pfeat)], 'PitchingPlus'] = pp_feat['PitchingPlus'].values

                pfeat['pitch_type']      = tm_sub['pitch_type'].values
                pfeat['plate_loc_side']  = tm_sub['plate_loc_side'].values
                pfeat['plate_loc_height']= tm_sub['plate_loc_height'].values
                pfeat['rel_speed']       = tm_sub['rel_speed'].values
                pfeat['induced_vert']    = tm_sub['induced_vert'].values
                pfeat['horz_break']      = tm_sub['horz_break'].values
                pfeat['spin_rate']       = tm_sub['spin_rate'].values

                plot_df = pfeat.dropna(subset=['plate_loc_side', 'plate_loc_height']).copy()

                PITCH_COLORS = {
                    'Fastball':  '#E53935',
                    'Sinker':    '#FF7043',
                    'Slider':    '#8E24AA',
                    'Curveball': '#1E88E5',
                    'ChangeUp':  '#43A047',
                    'Cutter':    '#FFB300',
                    'Splitter':  '#00ACC1',
                }

                fc1, fc2 = st.columns([2, 1])
                with fc1:
                    pt_options = sorted(plot_df['pitch_type'].unique())
                    selected_pts = st.multiselect(
                        "Show pitch types", pt_options, default=pt_options, key="pp_filter"
                    )
                with fc2:
                    color_by = st.selectbox(
                        "Color by", ["Pitch Type", "Stuff+", "Pitching+", "Velo"],
                        key="pp_color_by"
                    )

                plot_df = plot_df[plot_df['pitch_type'].isin(selected_pts)]

                fig_pp = go.Figure()

                if color_by == "Pitch Type":
                    for pt in selected_pts:
                        grp = plot_df[plot_df['pitch_type'] == pt]
                        if grp.empty: continue
                        fig_pp.add_trace(go.Scatter(
                            x=grp['plate_loc_side'],
                            y=grp['plate_loc_height'],
                            mode='markers',
                            name=pt,
                            marker=dict(
                                size=10,
                                color=PITCH_COLORS.get(pt, 'white'),
                                line=dict(color='rgba(255,255,255,0.3)', width=0.5),
                                opacity=0.85,
                            ),
                            customdata=np.stack([
                                grp['rel_speed'].round(1),
                                grp['induced_vert'].round(1),
                                grp['horz_break'].round(1),
                                grp['spin_rate'].round(0),
                                grp['StuffPlus'].round(1),
                                grp['PitchingPlus'].round(1) if grp['PitchingPlus'].notna().any() else np.full(len(grp), np.nan),
                            ], axis=-1),
                            hovertemplate=(
                                f"<b>{pt}</b><br>"
                                "Velo: %{customdata[0]} mph<br>"
                                "IVB: %{customdata[1]}in / HB: %{customdata[2]}in<br>"
                                "Spin: %{customdata[3]} rpm<br>"
                                "Stuff+: %{customdata[4]}<br>"
                                "Pitching+: %{customdata[5]}<br>"
                                "Side: %{x:.2f} ft · Height: %{y:.2f} ft"
                                "<extra></extra>"
                            ),
                        ))
                else:
                    color_col = {
                        'Stuff+':    'StuffPlus',
                        'Pitching+': 'PitchingPlus',
                        'Velo':      'rel_speed',
                    }[color_by]

                    colorscale = 'RdYlGn' if color_by != 'Velo' else 'Plasma'
                    cmin = plot_df[color_col].quantile(0.05) if color_col in plot_df else 85
                    cmax = plot_df[color_col].quantile(0.95) if color_col in plot_df else 100

                    fig_pp.add_trace(go.Scatter(
                        x=plot_df['plate_loc_side'],
                        y=plot_df['plate_loc_height'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=plot_df[color_col],
                            colorscale=colorscale,
                            cmin=cmin, cmax=cmax,
                            showscale=True,
                            colorbar=dict(title=color_by, thickness=15),
                            line=dict(color='rgba(255,255,255,0.3)', width=0.5),
                            opacity=0.9,
                        ),
                        text=plot_df['pitch_type'],
                        customdata=np.stack([
                            plot_df['rel_speed'].round(1),
                            plot_df['induced_vert'].round(1),
                            plot_df['horz_break'].round(1),
                            plot_df['spin_rate'].round(0),
                            plot_df['StuffPlus'].round(1),
                            plot_df['PitchingPlus'].round(1) if plot_df['PitchingPlus'].notna().any() else np.full(len(plot_df), np.nan),
                        ], axis=-1),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Velo: %{customdata[0]} mph<br>"
                            "IVB: %{customdata[1]}in / HB: %{customdata[2]}in<br>"
                            "Spin: %{customdata[3]} rpm<br>"
                            "Stuff+: %{customdata[4]}<br>"
                            "Pitching+: %{customdata[5]}<br>"
                            "Side: %{x:.2f} ft · Height: %{y:.2f} ft"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                fig_pp.add_trace(go.Scatter(
                    x=[-0.708, 0.708, 0.708, -0.708, -0.708],
                    y=[1.5, 1.5, 3.5, 3.5, 1.5],
                    mode='lines', line=dict(color='white', width=2.5),
                    hoverinfo='skip', showlegend=False
                ))
                for x in [-0.236, 0.236]:
                    fig_pp.add_shape(type='line', x0=x, x1=x, y0=1.5, y1=3.5,
                        line=dict(color='rgba(255,255,255,0.25)', width=1))
                for y in [2.167, 2.833]:
                    fig_pp.add_shape(type='line', x0=-0.708, x1=0.708, y0=y, y1=y,
                        line=dict(color='rgba(255,255,255,0.25)', width=1))
                fig_pp.add_trace(go.Scatter(
                    x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708],
                    y=[0.0, 0.0, 0.15, 0.22, 0.15, 0.0],
                    mode='lines', line=dict(color='lightgray', width=1.5),
                    hoverinfo='skip', showlegend=False
                ))

                fig_pp.update_layout(
                    xaxis=dict(
                        title="Horizontal (ft, pitcher's view)",
                        range=[-3.0, 3.0], zeroline=False, tickformat='.1f'
                    ),
                    yaxis=dict(
                        title="Height (ft)",
                        range=[-0.5, 5.5], zeroline=False,
                        scaleanchor='x', scaleratio=1
                    ),
                    height=620,
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='white'),
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1, font=dict(color='white', size=11)
                    ),
                    title=dict(
                        text=f"{pitcher_name} — Pitch Locations ({len(plot_df)} pitches)",
                        font=dict(size=14)
                    )
                )
                st.plotly_chart(fig_pp, use_container_width=True)
                st.caption("Pitcher's view  ·  Hover over any pitch for full details  ·  Strike zone divided into thirds")

    st.divider()
    if st.session_state.input_mode == "Arsenal Builder":
        if st.button("🗑️  Clear All Pitches", type="secondary"):
            st.session_state.pitches = []
            st.rerun()

else:
    if st.session_state.input_mode == "Arsenal Builder":
        st.info("👆  Select a pitch type and adjust the sliders above, then click **Add Pitch** to begin building your arsenal.")
    else:
        st.info("👈  Upload a TrackMan CSV in the sidebar to get started.")
