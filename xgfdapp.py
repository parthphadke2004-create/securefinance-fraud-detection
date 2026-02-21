import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureFinance — AI Fraud Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --bg-primary:    #F5F7FA;
    --bg-secondary:  #FFFFFF;
    --bg-card:       #FFFFFF;
    --bg-card-hover: #F0F4FF;
    --accent-gold:   #B8860B;
    --accent-teal:   #0D9E7E;
    --accent-red:    #D93025;
    --accent-blue:   #1A6FD4;
    --text-primary:  #0F172A;
    --text-secondary:#374151;
    --text-muted:    #6B7280;
    --border:        #E2E8F0;
    --border-bright: #CBD5E1;
    --success:       #0D9E7E;
    --danger:        #D93025;
    --warning:       #B8860B;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F7FA 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: var(--font-display) !important; }
h1 { font-size: 2.4rem !important; font-weight: 800 !important;
     color: #0F172A !important; letter-spacing: -0.5px; }
h2 { font-size: 1.6rem !important; font-weight: 700 !important;
     color: #0F172A !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important;
     color: #374151 !important; }
p, li { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }
label { color: var(--text-secondary) !important; font-size: 0.85rem !important;
        font-family: var(--font-body) !important; letter-spacing: 0.02em; }

/* ── Sidebar Brand ── */
.sf-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 0 20px 0; margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.sf-brand-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #C9A84C 0%, #A8873A 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.sf-brand-name {
    font-family: var(--font-display) !important;
    font-size: 1.15rem; font-weight: 800;
    color: #0F172A !important;
    line-height: 1.1;
}
.sf-brand-tagline {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem; color: var(--accent-gold) !important;
    letter-spacing: 0.12em; text-transform: uppercase;
}

/* ── Nav Radio Buttons ── */
div[data-testid="stRadio"] label {
    display: flex; align-items: center;
    padding: 10px 14px; border-radius: 8px;
    margin: 2px 0; cursor: pointer;
    transition: all 0.15s ease;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    border: 1px solid transparent;
}
div[data-testid="stRadio"] label:hover {
    background: #F0F4FF !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
div[data-testid="stRadio"] [aria-checked="true"] + label,
div[data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(201,168,76,0.12), transparent) !important;
    color: var(--accent-gold) !important;
    border-color: rgba(201,168,76,0.25) !important;
}

/* ── Cards ── */
.sf-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease;
}
.sf-card:hover { border-color: var(--border-bright); }
.sf-card-accent { border-left: 3px solid var(--accent-gold); }

/* ── Metric Cards ── */
.sf-metric {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.sf-metric-value {
    font-family: var(--font-display) !important;
    font-size: 2rem; font-weight: 800;
    color: var(--accent-gold);
    line-height: 1;
}
.sf-metric-label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem; color: var(--text-muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 6px;
}

/* ── Page Header Banner ── */
.sf-page-header {
    background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.sf-page-header::before {
    content: ''; position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(184,134,11,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.sf-page-title {
    font-family: var(--font-display) !important;
    font-size: 1.5rem; font-weight: 800;
    color: #0F172A !important;
}
.sf-page-subtitle {
    font-family: var(--font-body) !important;
    font-size: 0.88rem; color: #6B7280 !important;
    margin-top: 4px;
}
.sf-page-badge {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem; color: var(--accent-gold);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Prediction Result Boxes ── */
.sf-fraud-result {
    background: linear-gradient(135deg, rgba(255,77,109,0.1), rgba(255,77,109,0.04));
    border: 1px solid rgba(255,77,109,0.35);
    border-left: 4px solid var(--danger);
    border-radius: 12px;
    padding: 20px 24px;
    font-family: var(--font-display) !important;
    font-size: 1.3rem; font-weight: 800;
    color: var(--danger);
    letter-spacing: -0.3px;
}
.sf-legit-result {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(0,212,170,0.04));
    border: 1px solid rgba(0,212,170,0.35);
    border-left: 4px solid var(--success);
    border-radius: 12px;
    padding: 20px 24px;
    font-family: var(--font-display) !important;
    font-size: 1.3rem; font-weight: 800;
    color: var(--success);
    letter-spacing: -0.3px;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    color: var(--accent-gold) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #B8860B 0%, #96700A 100%) !important;
    color: #FFFFFF !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.2s ease !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* ── Inputs ── */
input, select, textarea,
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    color: #0F172A !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: var(--accent-gold) !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] div[role="slider"] {
    background: var(--accent-gold) !important;
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: #EEF2FF !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #6B7280 !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
}
div[aria-selected="true"][data-baseweb="tab"] {
    background: #FFFFFF !important;
    color: var(--accent-gold) !important;
    border: 1px solid var(--border-bright) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background: #FAFBFF !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 12px !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] { color: var(--accent-gold) !important; }

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background: rgba(184,134,11,0.06) !important;
    border: 1px solid rgba(184,134,11,0.2) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
}

/* ── Sidebar section label ── */
.sf-nav-label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem; font-weight: 500;
    color: #9CA3AF !important;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 16px 4px 6px 4px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Code blocks ── */
code, pre {
    background: #F1F5F9 !important;
    border: 1px solid var(--border) !important;
    color: #0D6E5F !important;
    font-family: var(--font-mono) !important;
    border-radius: 8px !important;
}

/* ── Status ribbon ── */
.sf-status-bar {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px;
    background: rgba(13,158,126,0.07);
    border: 1px solid rgba(13,158,126,0.2);
    border-radius: 8px;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem; color: #0D9E7E;
    margin-bottom: 16px;
}
.sf-status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #0D9E7E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;} 50%{opacity:0.35;}
}

/* ── Feature pill tags ── */
.sf-pill {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(26,111,212,0.08);
    border: 1px solid rgba(26,111,212,0.2);
    border-radius: 20px;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem; color: var(--accent-blue);
    margin: 2px;
}

/* ── Section separator ── */
.sf-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

/* Number input fix */
div[data-testid="stNumberInput"] input {
    font-family: var(--font-mono) !important;
}

/* Selectbox */
div[data-baseweb="select"] div {
    background: #FFFFFF !important;
    border-color: var(--border) !important;
    color: #0F172A !important;
    font-family: var(--font-mono) !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div {
    background: var(--accent-gold) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFBFF",
    font=dict(family="DM Sans, sans-serif", color="#374151"),
    title_font=dict(family="Syne, sans-serif", color="#0F172A", size=16),
    margin=dict(t=50, b=30, l=30, r=30),
    colorway=["#B8860B", "#0D9E7E", "#1A6FD4", "#D93025", "#7C3AED"],
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    return fig

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in [
    ("data", None), ("model", None), ("le_type", None),
    ("X_test", None), ("y_test", None), ("y_prob", None),
    ("metrics", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def engineer_features(df):
    d = df.copy()
    d["balanceDiff_Orig"]   = d["oldbalanceOrg"]  - d["newbalanceOrig"]
    d["balanceDiff_Dest"]   = d["newbalanceDest"]  - d["oldbalanceDest"]
    d["isOriginEmpty"]      = (d["newbalanceOrig"] == 0).astype(int)
    d["amountPercent_Orig"] = d["amount"] / (d["oldbalanceOrg"] + 1)
    d["errorBalanceOrig"]   = d["balanceDiff_Orig"] - d["amount"]
    d["errorBalanceDest"]   = d["balanceDiff_Dest"] - d["amount"]
    le = LabelEncoder()
    d["type_encoded"]       = le.fit_transform(d["type"])
    return d, le

FEATURE_COLS = [
    "step", "type_encoded", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "balanceDiff_Orig", "balanceDiff_Dest",
    "isOriginEmpty", "amountPercent_Orig", "errorBalanceOrig", "errorBalanceDest"
]

def page_header(badge, title, subtitle):
    st.markdown(f"""
    <div class="sf-page-header">
        <div class="sf-page-badge">{badge}</div>
        <div class="sf-page-title">{title}</div>
        <div class="sf-page-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sf-brand">
        <div class="sf-brand-icon">🔒</div>
        <div>
            <div class="sf-brand-name">SecureFinance</div>
            <div class="sf-brand-tagline">AI Fraud Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is not None:
        df_info = st.session_state.data
        fc = df_info["isFraud"].sum()
        st.markdown(f"""
        <div class="sf-status-bar">
            <div class="sf-status-dot"></div>
            Dataset active — {len(df_info):,} records
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sf-nav-label">Platform</div>', unsafe_allow_html=True)
    page = st.radio(
        "navigation",
        [
            "🏠  Overview",
            "📊  Data Intelligence",
            "📈  Analytics",
            "🤖  Model Training",
            "🔍  Transaction Scan",
            "📉  Performance Report"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#9CA3AF; line-height:1.8; padding: 0 4px;">
        <div>ENGINE &nbsp;&nbsp;&nbsp; XGBoost v2</div>
        <div>BALANCE &nbsp; SMOTE</div>
        <div>VERSION &nbsp; 3.1.0</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#B8860B; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">
            SecureFinance Platform
        </div>
        <h1 style="font-size:2.8rem !important; font-weight:900 !important; line-height:1.1 !important; color:#0F172A !important; margin:0 0 12px 0;">
            AI-Powered Fraud<br>
            <span style="color:#B8860B;">Intelligence Engine</span>
        </h1>
        <p style="font-size:1rem; color:#374151; max-width:560px; line-height:1.7; margin:0 0 32px 0;">
            Real-time transaction risk scoring powered by XGBoost gradient boosting.
            Detect fraud before it impacts your customers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row if data loaded
    if st.session_state.data is not None:
        df_s = st.session_state.data
        fc_s = df_s["isFraud"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",   f"{len(df_s):,}")
        c2.metric("Fraud Cases",     f"{int(fc_s):,}")
        c3.metric("Fraud Rate",      f"{fc_s/len(df_s)*100:.3f}%")
        model_status = "✓ Trained" if st.session_state.model else "Not Trained"
        c4.metric("Model Status",    model_status)
        st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="sf-card sf-card-accent">
            <h3 style="color:#B8860B !important; font-family:'Syne',sans-serif !important; font-size:0.8rem !important; letter-spacing:0.1em; text-transform:uppercase;">Platform Capabilities</h3>
            <div style="margin-top:16px; display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <span style="color:#B8860B; font-size:1.1rem;">◆</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#0F172A; font-size:0.92rem;">Data Intelligence</div>
                        <div style="font-size:0.82rem; color:#6B7280; margin-top:2px;">Upload and explore transaction datasets with rich statistical profiling</div>
                    </div>
                </div>
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <span style="color:#0D9E7E; font-size:1.1rem;">◆</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#0F172A; font-size:0.92rem;">XGBoost Model Engine</div>
                        <div style="font-size:0.82rem; color:#6B7280; margin-top:2px;">Gradient-boosted ensemble with SMOTE balancing for imbalanced fraud data</div>
                    </div>
                </div>
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <span style="color:#1A6FD4; font-size:1.1rem;">◆</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#0F172A; font-size:0.92rem;">Transaction Scan</div>
                        <div style="font-size:0.82rem; color:#6B7280; margin-top:2px;">Score any transaction in real-time with probability and risk gauge</div>
                    </div>
                </div>
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <span style="color:#D93025; font-size:1.1rem;">◆</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#0F172A; font-size:0.92rem;">Performance Report</div>
                        <div style="font-size:0.82rem; color:#6B7280; margin-top:2px;">ROC curves, confusion matrix, feature importance, and threshold tuning</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="sf-card" style="height:100%;">
            <h3 style="color:#B8860B !important; font-family:'Syne',sans-serif !important; font-size:0.8rem !important; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px;">Quick Start</h3>
            <div style="display:flex; flex-direction:column; gap:10px; font-family:'DM Mono',monospace; font-size:0.82rem;">
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="background:rgba(184,134,11,0.1); border:1px solid rgba(184,134,11,0.25); border-radius:6px; padding:2px 9px; color:#B8860B; font-weight:700;">01</span>
                    <span style="color:#374151;">Upload CSV dataset</span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="background:rgba(184,134,11,0.1); border:1px solid rgba(184,134,11,0.25); border-radius:6px; padding:2px 9px; color:#B8860B; font-weight:700;">02</span>
                    <span style="color:#374151;">Explore & visualize</span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="background:rgba(184,134,11,0.1); border:1px solid rgba(184,134,11,0.25); border-radius:6px; padding:2px 9px; color:#B8860B; font-weight:700;">03</span>
                    <span style="color:#374151;">Train XGBoost model</span>
                </div>
                <div style="display:flex; gap:10px; align-items:center;">
                    <span style="background:rgba(184,134,11,0.1); border:1px solid rgba(184,134,11,0.25); border-radius:6px; padding:2px 9px; color:#B8860B; font-weight:700;">04</span>
                    <span style="color:#374151;">Scan transactions</span>
                </div>
            </div>
            <div class="sf-sep" style="margin:20px 0;"></div>
            <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#9CA3AF; line-height:2;">
                <div>EXPECTED COLUMNS</div>
                <div style="color:#1A6FD4; margin-top:4px;">step · type · amount</div>
                <div style="color:#1A6FD4;">oldbalanceOrg · newbalanceOrig</div>
                <div style="color:#1A6FD4;">oldbalanceDest · newbalanceDest</div>
                <div style="color:#1A6FD4;">isFraud · isFlaggedFraud</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# DATA INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Intelligence":
    page_header("Data Intelligence", "Transaction Dataset Explorer",
                "Upload your fraud dataset to begin profiling and analysis")

    uploaded = st.file_uploader("Drop your CSV file here", type=["csv"],
                                 help="PaySim-format or similar transaction dataset")

    if uploaded:
        df = load_csv(uploaded)
        st.session_state.data = df
        st.success(f"✓ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

        tab1, tab2, tab3, tab4 = st.tabs([
            "  Preview  ", "  Statistics  ", "  Schema  ", "  Target Distribution  "
        ])

        with tab1:
            st.dataframe(df.head(25), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",          f"{df.shape[0]:,}")
            c2.metric("Columns",        df.shape[1])
            c3.metric("Missing Values", int(df.isnull().sum().sum()))
            c4.metric("Duplicates",     int(df.duplicated().sum()))

        with tab2:
            st.dataframe(df.describe().round(2), use_container_width=True)

        with tab3:
            buf = io.StringIO(); df.info(buf=buf)
            st.code(buf.getvalue(), language="text")
            miss = pd.DataFrame({
                "Column":    df.columns,
                "Dtype":     df.dtypes.values.astype(str),
                "Missing":   df.isnull().sum().values,
                "% Missing": (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(miss, use_container_width=True)

        with tab4:
            counts = df["isFraud"].value_counts()
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(values=counts.values, names=["Legitimate", "Fraudulent"],
                             title="Transaction Distribution",
                             color_discrete_sequence=["#0D9E7E", "#D93025"],
                             hole=0.55)
                apply_theme(fig)
                fig.update_traces(textfont_family="DM Mono, monospace")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(x=["Legitimate", "Fraudulent"], y=counts.values,
                             title="Transaction Count by Class",
                             color=["Legitimate", "Fraudulent"],
                             color_discrete_sequence=["#0D9E7E", "#D93025"])
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            c1.metric("Fraudulent",  f"{counts.get(1,0):,}")
            c2.metric("Legitimate",  f"{counts.get(0,0):,}")
            st.metric("Fraud Rate",  f"{counts.get(1,0)/len(df)*100:.4f}%")
    else:
        st.markdown("""
        <div class="sf-card" style="text-align:center; padding:60px; border-style:dashed;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📁</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#0F172A;">
                No dataset uploaded
            </div>
            <div style="font-size:0.85rem; color:#9CA3AF; margin-top:8px;">
                Use the file uploader above to load a CSV transaction dataset
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    page_header("Analytics", "Fraud Pattern Visualizations",
                "Explore transaction patterns and identify fraud signals")
    if st.session_state.data is None:
        st.warning("⚠️ Upload a dataset in Data Intelligence first.")
        st.stop()

    df = st.session_state.data
    viz = st.selectbox("Select Analysis", [
        "Transaction Type Breakdown",
        "Amount Distribution",
        "Balance Flow Analysis",
        "Feature Correlation Matrix"
    ])

    if viz == "Transaction Type Breakdown":
        c1, c2 = st.columns(2)
        with c1:
            tc = df["type"].value_counts()
            fig = px.bar(x=tc.index, y=tc.values, title="Volume by Transaction Type",
                         color=tc.values, color_continuous_scale=[[0,"#E2E8F0"],[1,"#B8860B"]])
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
        with c2:
            ft = df.groupby("type")["isFraud"].sum().sort_values(ascending=False)
            fig = px.bar(x=ft.index, y=ft.values, title="Fraud Incidents by Type",
                         color=ft.values, color_continuous_scale=[[0,"#FECACA"],[1,"#D93025"]])
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        # Fraud rate per type
        fraud_rate = df.groupby("type")["isFraud"].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=fraud_rate.index, y=fraud_rate.values,
                     title="Fraud Rate (%) by Transaction Type",
                     color=fraud_rate.values,
                     color_continuous_scale=[[0,"#0D9E7E"],[0.5,"#B8860B"],[1,"#D93025"]],
                     labels={"y":"Fraud Rate (%)"})
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    elif viz == "Amount Distribution":
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df["isFraud"]==0]["amount"],
                                       name="Legitimate", marker_color="#0D9E7E", opacity=0.7, nbinsx=60))
            fig.add_trace(go.Histogram(x=df[df["isFraud"]==1]["amount"],
                                       name="Fraudulent", marker_color="#D93025", opacity=0.7, nbinsx=60))
            fig.update_layout(title="Amount Distribution Overlay", barmode="overlay")
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="isFraud", y="amount", title="Amount by Fraud Status",
                         color="isFraud",
                         color_discrete_map={0:"#0D9E7E",1:"#D93025"},
                         labels={"isFraud":"Is Fraud","amount":"Transaction Amount"})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    elif viz == "Balance Flow Analysis":
        side = st.radio("Account Side", ["Origin", "Destination"], horizontal=True)
        c1, c2 = st.columns(2)
        if side == "Origin":
            with c1:
                samp = df.sample(min(8000, len(df)))
                fig = px.scatter(samp, x="oldbalanceOrg", y="newbalanceOrig", color="isFraud",
                                 title="Origin: Old vs New Balance",
                                 color_discrete_map={0:"#0D9E7E",1:"#D93025"}, opacity=0.55)
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, x="isFraud", y="oldbalanceOrg",
                             title="Origin Old Balance by Class",
                             color="isFraud", color_discrete_map={0:"#0D9E7E",1:"#D93025"})
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
        else:
            with c1:
                samp = df.sample(min(8000, len(df)))
                fig = px.scatter(samp, x="oldbalanceDest", y="newbalanceDest", color="isFraud",
                                 title="Destination: Old vs New Balance",
                                 color_discrete_map={0:"#0D9E7E",1:"#D93025"}, opacity=0.55)
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, x="isFraud", y="oldbalanceDest",
                             title="Destination Old Balance by Class",
                             color="isFraud", color_discrete_map={0:"#0D9E7E",1:"#D93025"})
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    elif viz == "Feature Correlation Matrix":
        df_fe, _ = engineer_features(df)
        num_cols = ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest",
                    "newbalanceDest","balanceDiff_Orig","balanceDiff_Dest","isOriginEmpty",
                    "amountPercent_Orig","errorBalanceOrig","errorBalanceDest","type_encoded","isFraud"]
        corr = df_fe[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        title="Feature Correlation Heatmap",
                        color_continuous_scale=[[0,"#D93025"],[0.5,"#F5F7FA"],[1,"#0D9E7E"]],
                        zmin=-1, zmax=1)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Training":
    page_header("Model Training", "Configure & Train XGBoost",
                "Tune hyperparameters and fit the fraud detection model with SMOTE balancing")
    if st.session_state.data is None:
        st.warning("⚠️ Upload a dataset in Data Intelligence first.")
        st.stop()

    df = st.session_state.data

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sf-card">', unsafe_allow_html=True)
        st.markdown("**⚖️ Data Splitting & Sampling**")
        test_size       = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        smote_strategy  = st.slider("SMOTE Strategy (minority ratio)", 0.1, 1.0, 0.5, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sf-card">', unsafe_allow_html=True)
        st.markdown("**🧠 XGBoost Hyperparameters**")
        n_estimators     = st.slider("n_estimators",        50, 500, 100, 50)
        max_depth        = st.slider("max_depth",             3, 12, 6)
        learning_rate    = st.slider("learning_rate",      0.01, 0.30, 0.10, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)

    c1b, c2b, c3b = st.columns(3)
    subsample        = c1b.slider("subsample",          0.5, 1.0, 0.8, 0.05)
    colsample_bytree = c2b.slider("colsample_bytree",   0.5, 1.0, 0.8, 0.05)
    scale_pos_weight = c3b.number_input("scale_pos_weight",  1, 300, 1,
                                        help="Increase to penalise missed fraud (class weight)")

    st.markdown("")
    if st.button("  🚀  Train XGBoost Model  ", type="primary"):
        with st.spinner("Engineering features · Applying SMOTE · Training XGBoost…"):
            df_fe, le_type = engineer_features(df)
            st.session_state.le_type = le_type

            X = df_fe[FEATURE_COLS]
            y = df_fe["isFraud"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            smote = SMOTE(random_state=42, sampling_strategy=smote_strategy)
            X_bal, y_bal = smote.fit_resample(X_train, y_train)

            model = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=colsample_bytree, scale_pos_weight=scale_pos_weight,
                random_state=42, n_jobs=-1, eval_metric="logloss",
                use_label_encoder=False
            )
            model.fit(X_bal, y_bal)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Precision":     precision_score(y_test, y_pred),
                "Recall":        recall_score(y_test, y_pred),
                "F1-Score":      f1_score(y_test, y_pred),
                "ROC-AUC":       roc_auc_score(y_test, y_prob),
                "CM":            confusion_matrix(y_test, y_pred),
                "Report":        classification_report(y_test, y_pred,
                                                        target_names=["Legitimate","Fraudulent"])
            }
            st.session_state.model   = model
            st.session_state.X_test  = X_test
            st.session_state.y_test  = y_test
            st.session_state.y_prob  = y_prob
            st.session_state.metrics = metrics

        st.success("✓ XGBoost trained successfully! Navigate to Performance Report for full results.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{metrics['Test Accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
        c3.metric("Recall",    f"{metrics['Recall']*100:.2f}%")
        c4.metric("F1-Score",  f"{metrics['F1-Score']*100:.2f}%")
        c5.metric("ROC-AUC",   f"{metrics['ROC-AUC']*100:.2f}%")

        st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)
        st.subheader("Feature Importances")
        imp = pd.DataFrame({"Feature": FEATURE_COLS,
                            "Importance": model.feature_importances_}
                           ).sort_values("Importance", ascending=True)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="XGBoost Feature Importances",
                     color="Importance",
                     color_continuous_scale=[[0,"#E2E8F0"],[1,"#B8860B"]])
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TRANSACTION SCAN
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Transaction Scan":
    page_header("Transaction Scan", "Real-Time Fraud Scoring",
                "Enter transaction details to receive an instant fraud risk assessment")
    if st.session_state.model is None:
        st.warning("⚠️ Train the XGBoost model first in Model Training.")
        st.stop()

    st.markdown('<div class="sf-card sf-card-accent">', unsafe_allow_html=True)
    st.markdown("**Transaction Parameters**")
    c1, c2, c3 = st.columns(3)
    with c1:
        step       = st.number_input("Step (time unit)", min_value=0, value=1)
        trans_type = st.selectbox("Transaction Type",
                                  ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount     = st.number_input("Amount (₹)", min_value=0.0, value=5000.0, format="%.2f")
    with c2:
        old_orig   = st.number_input("Old Balance — Origin",  min_value=0.0, value=20000.0, format="%.2f")
        new_orig   = st.number_input("New Balance — Origin",  min_value=0.0, value=15000.0, format="%.2f")
    with c3:
        old_dest   = st.number_input("Old Balance — Destination", min_value=0.0, value=1000.0, format="%.2f")
        new_dest   = st.number_input("New Balance — Destination", min_value=0.0, value=6000.0, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("  🔍  Analyze Transaction  ", type="primary"):
        le        = st.session_state.le_type
        type_enc  = le.transform([trans_type])[0]
        bd_o      = old_orig - new_orig
        bd_d      = new_dest - old_dest
        is_empty  = int(new_orig == 0)
        amt_pct   = amount / (old_orig + 1)
        err_o     = bd_o - amount
        err_d     = bd_d - amount

        feats = np.array([[step, type_enc, amount, old_orig, new_orig,
                           old_dest, new_dest, bd_o, bd_d, is_empty, amt_pct, err_o, err_d]])

        model  = st.session_state.model
        pred   = model.predict(feats)[0]
        prob   = model.predict_proba(feats)[0]

        st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)
        st.subheader("Risk Assessment")

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            if pred == 1:
                st.markdown('<div class="sf-fraud-result">🚨 &nbsp; HIGH RISK — FRAUD DETECTED</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="sf-legit-result">✅ &nbsp; LOW RISK — LEGITIMATE TRANSACTION</div>',
                            unsafe_allow_html=True)
        c2.metric("Fraud Probability",     f"{prob[1]*100:.2f}%")
        c3.metric("Confidence (Legit)",    f"{prob[0]*100:.2f}%")

        # Gauge
        bar_color = "#D93025" if prob[1] > 0.5 else "#0D9E7E"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob[1]*100, 2),
            title={"text": "FRAUD RISK SCORE", "font": {"family":"Syne,sans-serif","size":14,"color":"#6B7280"}},
            number={"suffix":"%", "font":{"family":"Syne,sans-serif","size":36,"color":bar_color}},
            gauge={
                "axis": {"range":[0,100], "tickcolor":"#CBD5E1", "tickfont":{"color":"#9CA3AF"}},
                "bar":  {"color": bar_color, "thickness":0.3},
                "bgcolor": "#FFFFFF",
                "bordercolor": "#E2E8F0",
                "steps":[
                    {"range":[0,30],   "color":"rgba(13,158,126,0.08)"},
                    {"range":[30,70],  "color":"rgba(184,134,11,0.08)"},
                    {"range":[70,100], "color":"rgba(217,48,37,0.1)"}
                ],
                "threshold":{"line":{"color":"#0F172A","width":2},"thickness":0.75,"value":50}
            }
        ))
        apply_theme(fig)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        # Engineered features breakdown
        with st.expander("🔬 View Engineered Features Used for Prediction"):
            feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Value": feats[0]})
            st.dataframe(feat_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "📉  Performance Report":
    page_header("Performance Report", "XGBoost Model Evaluation",
                "Comprehensive metrics, visualizations, and threshold sensitivity analysis")
    if st.session_state.metrics is None:
        st.warning("⚠️ Train the XGBoost model first in Model Training.")
        st.stop()

    m   = st.session_state.metrics
    y_t = st.session_state.y_test
    y_p = st.session_state.y_prob

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['Test Accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['Precision']*100:.2f}%")
    c3.metric("Recall",    f"{m['Recall']*100:.2f}%")
    c4.metric("F1-Score",  f"{m['F1-Score']*100:.2f}%")
    c5.metric("ROC-AUC",   f"{m['ROC-AUC']*100:.2f}%")

    st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm  = m["CM"]
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Legitimate","Fraudulent"], y=["Legitimate","Fraudulent"],
                        color_continuous_scale=[[0,"#EEF2FF"],[1,"#B8860B"]],
                        title="Confusion Matrix")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_t, y_p)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", fill="tozeroy",
            fillcolor="rgba(184,134,11,0.07)",
            name=f"XGBoost  AUC = {m['ROC-AUC']:.4f}",
            line=dict(color="#B8860B", width=2.5)
        ))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="Random Baseline",
                                 line=dict(color="#CBD5E1", dash="dash", width=1.5)))
        fig.update_layout(title="Receiver Operating Characteristic",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.subheader("Classification Report")
    st.code(m["Report"], language="text")

    st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)

    # Feature Importances
    st.subheader("Feature Importances")
    model = st.session_state.model
    imp   = pd.DataFrame({"Feature": FEATURE_COLS,
                          "Importance": model.feature_importances_}
                         ).sort_values("Importance", ascending=True)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title="XGBoost Feature Importances — Gain",
                 color="Importance",
                 color_continuous_scale=[[0,"#E2E8F0"],[0.5,"#1A6FD4"],[1,"#B8860B"]])
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sf-sep"></div>', unsafe_allow_html=True)

    # Threshold Tuning
    st.subheader("🎚️ Decision Threshold Tuning")
    st.markdown("""
    <div style="font-size:0.83rem; color:#374151; margin-bottom:16px;">
        Adjust the classification threshold to trade off between <span style="color:#B8860B;">Precision</span>
        and <span style="color:#0D9E7E;">Recall</span> based on your business requirements.
        Lower threshold = catch more fraud (higher recall, lower precision).
    </div>
    """, unsafe_allow_html=True)

    threshold  = st.slider("Classification Threshold", 0.01, 0.99, 0.50, 0.01)
    y_pred_t   = (y_p >= threshold).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision_score(y_t,y_pred_t,zero_division=0)*100:.2f}%")
    c2.metric("Recall",    f"{recall_score(y_t,y_pred_t,zero_division=0)*100:.2f}%")
    c3.metric("F1-Score",  f"{f1_score(y_t,y_pred_t,zero_division=0)*100:.2f}%")
    c4.metric("Accuracy",  f"{accuracy_score(y_t,y_pred_t)*100:.2f}%")

    # Precision-Recall curve for context
    from sklearn.metrics import precision_recall_curve
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_t, y_p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines",
                             name="Precision-Recall",
                             line=dict(color="#0D9E7E", width=2.5)))
    fig.add_vline(x=recall_score(y_t,y_pred_t,zero_division=0),
                  line=dict(color="#B8860B", dash="dot", width=1.5),
                  annotation_text=f"Threshold {threshold:.2f}",
                  annotation_font_color="#B8860B")
    fig.update_layout(title="Precision vs Recall Curve",
                      xaxis_title="Recall", yaxis_title="Precision")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)