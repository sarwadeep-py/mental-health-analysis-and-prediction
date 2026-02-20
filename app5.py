"""
MindPulse v3 — Mental Health Analysis Dashboard (Fixed)
========================================================
Run:  streamlit run fixed_app.py

Requirements:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import warnings, random, io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MindPulse", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

BG     = "#0b1120"
CARD   = "#111827"
CARD2  = "#161f30"
BORDER = "#1e2d42"
TEAL   = "#00c9a7"
AMBER  = "#f5a623"
ROSE   = "#f95f7a"
BLUE   = "#4facfe"
PURPLE = "#a78bfa"
TEXT   = "#e2e8f0"
MUTED  = "#8896ae"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{BG}; color:{TEXT}; }}
h1,h2,h3 {{ font-family:'Syne',sans-serif; font-weight:800; }}
.main .block-container {{ background:{BG}; padding-top:1.5rem; max-width:1400px; }}
section[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:20px; margin-bottom:14px; }}
.kpi {{ text-align:center; padding:16px 8px; }}
.kpi-val {{ font-family:'Syne',sans-serif; font-size:30px; font-weight:800; }}
.kpi-lbl {{ font-size:12px; color:{MUTED}; margin-top:3px; }}
.sh {{ font-family:'Syne',sans-serif; font-size:17px; font-weight:700; color:{TEXT};
       border-left:4px solid {TEAL}; padding-left:10px; margin:20px 0 12px; }}
.pct-outer {{ background:{BORDER}; border-radius:100px; height:32px; overflow:hidden; margin:8px 0 3px; }}
.pct-inner {{ height:100%; border-radius:100px; display:flex; align-items:center;
              justify-content:flex-end; padding-right:12px; font-size:14px; font-weight:700; color:#fff; }}
.pct-ticks {{ display:flex; justify-content:space-between; font-size:10px; color:{MUTED}; padding:0 3px; margin-bottom:4px; }}
.big-pct {{ font-family:'Syne',sans-serif; font-size:80px; font-weight:800; line-height:1; text-align:center; margin:8px 0; }}
.rec {{ border-radius:12px; padding:14px 18px; margin-bottom:10px; border-left:4px solid {TEAL};
        background:linear-gradient(135deg,{CARD},{BG}); }}
.rec.amber {{ border-left-color:{AMBER}; }}
.rec.rose  {{ border-left-color:{ROSE};  }}
.rec.purple{{ border-left-color:{PURPLE}; }}
.rec-title {{ font-weight:600; font-size:14px; margin-bottom:5px; }}
.rec-body  {{ font-size:13px; color:{MUTED}; line-height:1.65; }}
.thought {{ background:linear-gradient(135deg,#0d2a24,#0b1120); border:1px solid {TEAL}44;
            border-radius:14px; padding:18px 22px; margin-bottom:10px;
            font-size:14px; font-style:italic; color:{TEXT}; line-height:1.7; }}
.q-label {{ font-size:14px; font-weight:600; color:{TEXT}; margin-bottom:6px; margin-top:16px; }}
button[data-baseweb="tab"] {{ font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:14px !important; }}
hr.div {{ border:none; border-top:1px solid {BORDER}; margin:18px 0; }}
</style>
""", unsafe_allow_html=True)


# ─── FIX 1: render matplotlib to buffer, return image ─────────────────────────
def render_fig(fig):
    """Render matplotlib figure to PNG buffer and display — avoids blank chart bug."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def dark_fig(w=5, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CARD)
    ax.set_facecolor(CARD2)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT); ax.grid(color=BORDER, alpha=0.5, linewidth=0.5)
    return fig, ax


# ─── FIX 2: load data correctly for both file path and uploaded file ───────────
@st.cache_data
def load_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_data(src):
    if src is None:
        return load_from_path("student_depression_dataset.csv")
    if isinstance(src, str):
        return load_from_path(src)
    # Uploaded file — read bytes, reset, parse
    src.seek(0)
    return pd.read_csv(src)


@st.cache_resource
def train_model(df_bytes: bytes):
    """Accept bytes so cache key works correctly with uploaded files."""
    import io as _io
    _df = pd.read_csv(_io.BytesIO(df_bytes))
    FEAT = ["Sleep Duration","Dietary Habits",
            "Have you ever had suicidal thoughts ?",
            "Work/Study Hours","Financial Stress",
            "Family History of Mental Illness"]
    data = _df[FEAT+["Depression"]].copy()
    data.replace("?", np.nan, inplace=True); data.dropna(inplace=True)
    le_map = {}
    for c in ["Sleep Duration","Dietary Habits",
              "Have you ever had suicidal thoughts ?",
              "Family History of Mental Illness"]:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].astype(str).str.strip("'"))
        le_map[c] = le
    data["Financial Stress"] = pd.to_numeric(data["Financial Stress"], errors="coerce")
    data["Work/Study Hours"] = pd.to_numeric(data["Work/Study Hours"], errors="coerce")
    data.dropna(inplace=True)
    X, y = data[FEAT], data["Depression"]
    Xtr,_,ytr,_ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr)
    mdl = AdaBoostClassifier(n_estimators=100, random_state=42); mdl.fit(Xtr_s, ytr)
    return mdl, sc, le_map


def predict_prob(mdl, sc, le_map, ans):
    FEAT_ORDER = ["Sleep Duration","Dietary Habits",
                  "Have you ever had suicidal thoughts ?",
                  "Work/Study Hours","Financial Stress",
                  "Family History of Mental Illness"]
    CAT = {"Sleep Duration","Dietary Habits",
           "Have you ever had suicidal thoughts ?","Family History of Mental Illness"}
    row = []
    for c in FEAT_ORDER:
        if c in CAT:
            v = str(ans[c]).strip("'")
            le = le_map[c]
            row.append(int(le.transform([v])[0]) if v in le.classes_ else 0)
        else:
            row.append(float(ans[c]))
    return float(mdl.predict_proba(sc.transform([row]))[0][1])


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='font-family:Syne;color:{TEAL}'>🧠 MindPulse</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
try:
    if uploaded:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
        uploaded.seek(0)
        df_bytes = uploaded.read()
    else:
        df = load_from_path("student_depression_dataset.csv")
        with open("student_depression_dataset.csv", "rb") as f:
            df_bytes = f.read()
except FileNotFoundError:
    st.error("Dataset not found. Place `student_depression_dataset.csv` in the same folder or upload via sidebar.")
    st.stop()

try:
    mdl, sc, le_map = train_model(df_bytes)
    model_ok = True
except Exception as e:
    model_ok = False
    st.sidebar.warning(f"Model error: {e}")


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:18px 0 10px'>
  <h1 style='font-size:38px;margin:0;background:linear-gradient(90deg,{TEAL},{BLUE});
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:Syne'>
    🧠 MindPulse
  </h1>
  <p style='color:{MUTED};font-size:15px;margin:4px 0 0'>
    Student Mental Health · Analytics &amp; Depression Risk Assessment
  </p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
total    = len(df)
dep_c    = int(df["Depression"].sum()) if "Depression" in df.columns else 0
dep_rate = round(dep_c/total*100,1) if total else 0
avg_hrs  = round(pd.to_numeric(df["Work/Study Hours"], errors="coerce").mean(),1)
avg_age  = round(pd.to_numeric(df["Age"], errors="coerce").mean(),1)

for col, lbl, val, clr in zip(st.columns(5),
    ["Total Students","Depressed","Depression Rate","Avg Study Hrs","Avg Age"],
    [f"{total:,}", f"{dep_c:,}", f"{dep_rate}%", f"{avg_hrs}h", f"{avg_age} yrs"],
    [BLUE, ROSE, AMBER, TEAL, PURPLE]):
    col.markdown(f"""
    <div class='card kpi'>
      <div class='kpi-val' style='color:{clr}'>{val}</div>
      <div class='kpi-lbl'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  Dashboard", "🩺  Assessment", "📈  Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sh'>🎛️ Interactive Filters</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        g_opts = ["All"] + (sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else [])
        sel_g  = st.selectbox("Gender", g_opts)
    with f2:
        sel_s  = st.selectbox("Sleep Duration", ["All","Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"])
    with f3:
        sel_f  = st.selectbox("Financial Stress", ["All","1","2","3","4","5"])
    with f4:
        sel_d  = st.selectbox("Depression Status", ["All","Depressed","Not Depressed"])

    dff = df.copy()
    # Normalize string columns — strip quotes that may exist in raw CSV values
    dff["Sleep Duration"] = dff["Sleep Duration"].astype(str).str.strip("'").str.strip()
    dff["Dietary Habits"] = dff["Dietary Habits"].astype(str).str.strip("'").str.strip()
    dff["Have you ever had suicidal thoughts ?"] = dff["Have you ever had suicidal thoughts ?"].astype(str).str.strip("'").str.strip()
    dff["Family History of Mental Illness"] = dff["Family History of Mental Illness"].astype(str).str.strip("'").str.strip()
    dff["Financial Stress"] = pd.to_numeric(dff["Financial Stress"].astype(str).str.strip("'"), errors="coerce")

    if sel_g != "All" and "Gender" in dff.columns:
        dff = dff[dff["Gender"] == sel_g]
    if sel_s != "All":
        dff = dff[dff["Sleep Duration"] == sel_s]
    if sel_f != "All":
        dff = dff[dff["Financial Stress"] == float(sel_f)]
    if sel_d == "Depressed":
        dff = dff[dff["Depression"] == 1]
    elif sel_d == "Not Depressed":
        dff = dff[dff["Depression"] == 0]

    if len(dff) == 0:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        st.stop()

    fdep  = int(dff["Depression"].sum()) if len(dff) else 0
    frate = round(fdep/len(dff)*100,1) if len(dff) else 0
    fk1, fk2, fk3 = st.columns(3)
    fk1.markdown(f"<div class='card kpi'><div class='kpi-val' style='color:{BLUE}'>{len(dff):,}</div><div class='kpi-lbl'>Filtered Records</div></div>", unsafe_allow_html=True)
    fk2.markdown(f"<div class='card kpi'><div class='kpi-val' style='color:{ROSE}'>{fdep:,}</div><div class='kpi-lbl'>Depressed</div></div>", unsafe_allow_html=True)
    fk3.markdown(f"<div class='card kpi'><div class='kpi-val' style='color:{AMBER}'>{frate}%</div><div class='kpi-lbl'>Depression Rate</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>📊 Distribution Charts</div>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        fig, ax = dark_fig()
        age_data = pd.to_numeric(dff["Age"], errors="coerce").dropna()
        ax.hist(age_data, bins=20, color=BLUE, alpha=0.8, edgecolor=CARD)
        ax.axvline(age_data.median(), color=AMBER, ls="--", lw=1.5, label=f"Median: {age_data.median():.0f}")
        ax.set_title("Age Distribution", fontweight="bold")
        ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend(fontsize=8, framealpha=0.2)
        render_fig(fig)

    with r1c2:
        so = ["Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"]
        sv = dff["Sleep Duration"].value_counts().reindex(so, fill_value=0)
        fig, ax = dark_fig()
        bar_colors = [ROSE, AMBER, TEAL, BLUE]
        bars = ax.bar(["<5h","5-6h","7-8h",">8h"], sv.values,
                      color=bar_colors, edgecolor=CARD, alpha=0.85, width=0.6)
        max_val = max(sv.values) if max(sv.values) > 0 else 1
        ax.set_ylim(0, max_val * 1.18)
        for b, v in zip(bars, sv.values):
            if v > 0:
                ax.text(b.get_x() + b.get_width()/2, v + max_val * 0.03,
                        f"{v:,}", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_title("Sleep Duration", fontweight="bold"); ax.set_ylabel("Count")
        render_fig(fig)

    with r1c3:
        dc = dff["Dietary Habits"].value_counts()
        if len(dc) > 0:
            fig, ax = dark_fig()
            clrs = [TEAL,AMBER,ROSE,BLUE][:len(dc)]
            wedges,txts,atxts = ax.pie(dc.values, labels=dc.index, colors=clrs,
                autopct="%1.1f%%", pctdistance=0.75, startangle=90,
                wedgeprops={"edgecolor":CARD,"linewidth":2})
            for t in txts+atxts: t.set_color(TEXT); t.set_fontsize(9)
            ax.set_title("Dietary Habits", fontweight="bold")
            render_fig(fig)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        if "Gender" in dff.columns:
            gd = dff.groupby("Gender")["Depression"].value_counts().unstack(fill_value=0)
            fig, ax = dark_fig()
            x,w = np.arange(len(gd)),0.35
            ax.bar(x-w/2, gd.get(0,0), w, label="No Depression", color=TEAL, alpha=0.85)
            ax.bar(x+w/2, gd.get(1,0), w, label="Depression",    color=ROSE, alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(gd.index)
            ax.set_title("Depression by Gender", fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.2)
            render_fig(fig)

    with r2c2:
        fr = dff.groupby("Financial Stress")["Depression"].mean()*100
        if len(fr) > 0:
            fig, ax = dark_fig()
            bars = ax.bar(fr.index.astype(str), fr.values, color=AMBER, alpha=0.85, edgecolor=CARD)
            for b,v in zip(bars,fr.values):
                ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.0f}%",
                        ha="center", fontsize=8, color=TEXT)
            ax.set_title("Financial Stress → Depression %", fontweight="bold")
            ax.set_xlabel("Financial Stress (1-5)"); ax.set_ylabel("Depression %")
            render_fig(fig)

    with r2c3:
        si = dff["Have you ever had suicidal thoughts ?"].value_counts()
        fig, ax = dark_fig()
        ax.bar(si.index, si.values, color=[TEAL,ROSE][:len(si)], edgecolor=CARD, alpha=0.85)
        ax.set_title("Suicidal Thoughts\n★ Strongest Predictor", fontweight="bold"); ax.set_ylabel("Count")
        render_fig(fig)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fh = dff.groupby("Family History of Mental Illness")["Depression"].value_counts().unstack(fill_value=0)
        fig, ax = dark_fig(6,4)
        x,w = np.arange(len(fh)),0.35
        ax.bar(x-w/2, fh.get(0,0), w, label="No Depression", color=TEAL, alpha=0.85)
        ax.bar(x+w/2, fh.get(1,0), w, label="Depression",    color=ROSE, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(fh.index)
        ax.set_title("Depression vs Family History", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.2)
        render_fig(fig)

    with r3c2:
        dv = dff["Depression"].value_counts()
        if len(dv) > 0:
            fig, ax = dark_fig(6,4)
            ax.pie(dv.values,
                   labels=[("No Depression" if i==0 else "Depressed") for i in dv.index],
                   colors=[TEAL,ROSE][:len(dv)],
                   autopct="%1.1f%%", startangle=90, pctdistance=0.75,
                   wedgeprops={"edgecolor":CARD,"linewidth":3})
            for t in ax.texts: t.set_color(TEXT); t.set_fontsize(10)
            ax.set_title("Overall Depression Rate", fontweight="bold")
            render_fig(fig)

    # Plotly scatter — fully interactive
    st.markdown("<div class='sh'>🔬 Interactive Scatter — Financial Stress vs Study Hours</div>", unsafe_allow_html=True)
    sc_df = dff[["Financial Stress","Work/Study Hours","Age","Depression","Dietary Habits"]].copy()
    sc_df["Work/Study Hours"] = pd.to_numeric(sc_df["Work/Study Hours"], errors="coerce")
    sc_df["Status"] = sc_df["Depression"].map({0:"No Depression",1:"Depressed"})
    sc_df.dropna(subset=["Financial Stress","Work/Study Hours"], inplace=True)
    if len(sc_df) > 0:
        fig_sc = px.scatter(sc_df.sample(min(2000, len(sc_df)), random_state=42),
            x="Financial Stress", y="Work/Study Hours", color="Status",
            color_discrete_map={"No Depression":TEAL,"Depressed":ROSE},
            opacity=0.65, hover_data=["Age","Dietary Habits"],
            title="Financial Stress vs Study Hours (coloured by Depression)")
        fig_sc.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD2, font_color=TEXT,
            title_font_family="Syne", legend_title_text="", height=380,
            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='sh'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    nc = [c for c in ["Age","Academic Pressure","CGPA","Study Satisfaction",
                       "Work/Study Hours","Financial Stress","Depression"] if c in dff.columns]
    corr = dff[nc].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(11,4.5), facecolor=CARD)
    ax.set_facecolor(CARD)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, linecolor=BG, annot_kws={"size":9,"color":TEXT},
                cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    render_fig(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class='card' style='border-left:4px solid {BLUE};padding:16px 20px;margin-bottom:20px'>
      <b style='font-size:15px'>📋 Mental Health Self-Assessment</b><br>
      <span style='color:{MUTED};font-size:13px'>
        Answer all 6 questions honestly. Our AdaBoost model will estimate your depression risk.
        This is <b style='color:{TEXT}'>not a medical diagnosis</b>.
      </span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("assessment"):
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("<div class='q-label'>💤 Q1 · How many hours do you sleep daily?</div>", unsafe_allow_html=True)
            q_sleep = st.select_slider("sleep", options=["Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"],
                                       value="7-8 hours", label_visibility="collapsed")
            st.markdown("<div class='q-label'>🥗 Q2 · How would you describe your dietary habits?</div>", unsafe_allow_html=True)
            q_diet  = st.radio("diet", ["Healthy","Moderate","Unhealthy"], horizontal=True, label_visibility="collapsed")
            st.markdown("<div class='q-label'>📚 Q3 · How many hours do you study/work daily?</div>", unsafe_allow_html=True)
            q_hrs   = st.slider("hrs", 0, 16, 6, label_visibility="collapsed")
        with q2:
            st.markdown("<div class='q-label'>💰 Q4 · Financial stress level (1=low, 5=high)</div>", unsafe_allow_html=True)
            q_fin   = st.slider("fin", 1, 5, 2, label_visibility="collapsed")
            st.markdown("<div class='q-label'>💭 Q5 · Have you ever had suicidal thoughts?</div>", unsafe_allow_html=True)
            q_sui   = st.radio("sui", ["No","Yes"], horizontal=True, label_visibility="collapsed")
            if q_sui == "Yes":
                st.markdown(f"<div style='background:#2a1018;border:1px solid {ROSE};border-radius:8px;padding:10px 14px;font-size:12px;color:{ROSE}'>Help is available: <b>iCall 9152987821</b> · <b>Vandrevala 1860-2662-345</b> (24/7)</div>", unsafe_allow_html=True)
            st.markdown("<div class='q-label'>🧬 Q6 · Family history of mental illness?</div>", unsafe_allow_html=True)
            q_fam   = st.radio("fam", ["No","Yes"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Analyse My Responses", use_container_width=True, type="primary")

    if submitted and model_ok:
        ans = {
            "Sleep Duration":                        q_sleep,
            "Dietary Habits":                        q_diet,
            "Have you ever had suicidal thoughts ?": q_sui,
            "Work/Study Hours":                      q_hrs,
            "Financial Stress":                      q_fin,
            "Family History of Mental Illness":      q_fam,
        }
        st.session_state["result"] = {"pct": round(predict_prob(mdl, sc, le_map, ans)*100, 1), "ans": ans}

    if submitted and not model_ok:
        st.error("Model not ready — check your dataset.")

    if "result" in st.session_state:
        pct = st.session_state["result"]["pct"]
        ans = st.session_state["result"]["ans"]

        if pct < 35:    clr, lbl, bg = TEAL,  "Low Risk",      "#0d2a24"
        elif pct < 60:  clr, lbl, bg = AMBER, "Moderate Risk",  "#2a2110"
        else:           clr, lbl, bg = ROSE,  "High Risk",      "#2a1018"

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>🎯 Your Result</div>", unsafe_allow_html=True)

        rl, rr = st.columns([1,1])
        with rl:
            st.markdown(f"""
            <div style='background:{bg};border:2px solid {clr};border-radius:18px;padding:30px 26px;text-align:center'>
              <div style='color:{MUTED};font-size:12px;letter-spacing:2px;text-transform:uppercase'>Depression Risk Score</div>
              <div class='big-pct' style='color:{clr}'>{pct}%</div>
              <div style='font-family:Syne;font-size:20px;font-weight:700;color:{clr};margin-bottom:18px'>{lbl}</div>
              <div class='pct-outer'>
                <div class='pct-inner' style='width:{pct}%;background:linear-gradient(90deg,{clr}55,{clr})'>{pct}%</div>
              </div>
              <div class='pct-ticks'><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
              <div style='margin-top:14px;display:flex;justify-content:space-around;font-size:12px'>
                <span>🟢 Low &lt;35%</span><span>🟡 Moderate 35–60%</span><span>🔴 High &gt;60%</span>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            rp1, rp2 = st.columns(2)
            for i,(l2,v) in enumerate([
                ("💤 Sleep", ans["Sleep Duration"]),
                ("🥗 Diet",  ans["Dietary Habits"]),
                ("📚 Study Hrs", f"{int(ans['Work/Study Hours'])} hrs/day"),
                ("💰 Fin. Stress", f"{int(ans['Financial Stress'])}/5"),
                ("💭 Suicidal Thts", ans["Have you ever had suicidal thoughts ?"]),
                ("🧬 Family Hist.", ans["Family History of Mental Illness"]),
            ]):
                (rp1 if i%2==0 else rp2).markdown(f"""
                <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                            padding:10px 14px;margin-bottom:8px;font-size:13px'>
                  <span style='color:{MUTED}'>{l2}</span><br><b style='color:{TEXT}'>{v}</b>
                </div>""", unsafe_allow_html=True)

        with rr:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"color":TEXT,"size":44,"family":"Syne"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":TEXT,"tickfont":{"color":TEXT,"size":10}},
                    "bar":{"color":clr,"thickness":0.3}, "bgcolor":CARD, "bordercolor":BORDER,
                    "steps":[{"range":[0,35],"color":"#0d2a24"},
                              {"range":[35,60],"color":"#2a2110"},
                              {"range":[60,100],"color":"#2a1018"}],
                    "threshold":{"line":{"color":"white","width":3},"thickness":0.85,"value":pct}
                },
                title={"text":"Risk Gauge","font":{"color":TEXT,"size":15,"family":"Syne"}}
            ))
            fig_g.update_layout(paper_bgcolor=CARD, font_color=TEXT,
                                margin=dict(l=30,r=30,t=60,b=10), height=320)
            st.plotly_chart(fig_g, use_container_width=True)

            sleep_map = {"Less than 5 hours":3.5,"5-6 hours":5.5,"7-8 hours":7.5,"More than 8 hours":9}
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="You",
                x=["Sleep Hrs","Study Hrs","Fin.Stress×1.6"],
                y=[sleep_map.get(ans["Sleep Duration"],6), float(ans["Work/Study Hours"]), float(ans["Financial Stress"])*1.6],
                marker_color=clr, opacity=0.9))
            fig_bar.add_trace(go.Bar(name="Avg Student",
                x=["Sleep Hrs","Study Hrs","Fin.Stress×1.6"],
                y=[df["Sleep Duration"].astype(str).str.strip("'").map(sleep_map).mean(),
                   pd.to_numeric(df["Work/Study Hours"],errors="coerce").mean(),
                   pd.to_numeric(df["Financial Stress"],errors="coerce").mean()*1.6],
                marker_color=MUTED, opacity=0.6))
            fig_bar.update_layout(barmode="group", paper_bgcolor=CARD, plot_bgcolor=CARD2,
                font_color=TEXT, height=230, margin=dict(l=10,r=10,t=10,b=30),
                legend=dict(font=dict(size=11)), xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>💡 Personalised Recommendations</div>", unsafe_allow_html=True)

        recs = []
        if pct >= 60:
            recs.append(("rose","🏥 Seek Professional Support",
                "Your risk score is high. Please consider speaking with a licensed therapist. Professional support can make a significant difference. You don't have to face this alone."))
        if ans["Have you ever had suicidal thoughts ?"] == "Yes":
            recs.append(("rose","🆘 Immediate Support Is Available",
                "Help is available right now — iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7). Please reach out today."))
        if ans["Sleep Duration"] in ["Less than 5 hours","5-6 hours"]:
            recs.append(("amber","💤 Improve Your Sleep",
                "Aim for 7–9 hours: fixed bedtime, no screens 1 hr before bed, no caffeine after 3 PM. Even 30 extra minutes shifts your mood noticeably."))
        if ans["Dietary Habits"] == "Unhealthy":
            recs.append(("amber","🥗 Nourish Your Brain",
                "Omega-3 foods (fish, walnuts, flaxseed), leafy greens, and whole grains support serotonin. Reduce processed food and sugar — both amplify mood instability."))
        if int(ans["Financial Stress"]) >= 4:
            recs.append(("amber","💰 Tackle Financial Anxiety",
                "Break it into small steps: speak with a student advisor, explore scholarships, or use a budgeting app. Writing it down reduces mental overwhelm."))
        if int(ans["Work/Study Hours"]) >= 10:
            recs.append(("amber","📚 Protect Recovery Time",
                f"Studying {int(ans['Work/Study Hours'])} hrs/day risks burnout. Try Pomodoro (25 min on, 5 off) and schedule one full rest day per week."))
        if ans["Family History of Mental Illness"] == "Yes":
            recs.append(("teal","🧬 Proactive Mental Health Checks",
                "Family history increases baseline risk. Regular check-ins and a strong support network are key preventive strategies."))
        recs.append(("teal","🏃 Move Your Body",
            "30 min of moderate exercise 5× per week reduces depression comparably to medication in mild cases. A 15-min walk or yoga counts."))
        recs.append(("teal","🧘 Mindfulness & Social Connection",
            "5–10 min of daily mindfulness reduces rumination. Prioritise at least one meaningful social interaction daily."))
        recs.append(("purple","📖 Track Your Mood",
            "A one-sentence daily journal reveals hidden patterns. Apps like Daylio make it effortless. Awareness is the first step to change."))

        rc1, rc2 = st.columns(2)
        for i,(style,title,body) in enumerate(recs):
            (rc1 if i%2==0 else rc2).markdown(f"""
            <div class='rec {style}'>
              <div class='rec-title'>{title}</div>
              <div class='rec-body'>{body}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>🌱 Affirmations for You</div>", unsafe_allow_html=True)
        all_thoughts = [
            "\"You are not your worst day. Every small step forward — even just getting up — is a victory worth acknowledging.\"",
            "\"Asking for help is one of the bravest things a person can do. It takes more strength to reach out than to stay silent.\"",
            "\"Your mental health matters just as much as your grades. A mind at rest learns better, thinks clearer, and lives fuller.\"",
            "\"Progress is not always visible. Seeds grow underground before they break the surface — so does healing.\"",
            "\"You have survived 100% of your hardest days so far. That is not nothing. That is everything.\"",
            "\"Rest is not laziness. It is the foundation on which resilience is built. You are allowed to rest.\"",
            "\"One difficult chapter does not define your story. The pen is still in your hand.\"",
            "\"Being kind to yourself in hard moments is not weakness — it is the most productive thing you can do.\"",
        ]
        random.seed(int(pct * 13))
        tc1, tc2, tc3 = st.columns(3)
        for col, t in zip([tc1,tc2,tc3], random.sample(all_thoughts, 3)):
            col.markdown(f"<div class='thought'>{t}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                    padding:12px 16px;margin-top:6px;font-size:12px;color:{MUTED}'>
        ⚠️ <b style='color:{TEXT}'>Disclaimer:</b> For educational purposes only. Not a medical diagnosis.
        If distressed, contact a qualified mental health professional.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sh'>📈 Advanced Risk Analytics</div>", unsafe_allow_html=True)

    ai1, ai2 = st.columns(2)
    with ai1:
        df["_stress"] = (pd.to_numeric(df.get("Academic Pressure",0),errors="coerce").fillna(0) +
                         pd.to_numeric(df["Financial Stress"],errors="coerce").fillna(0))
        sd = df.groupby("_stress")["Depression"].mean()*100
        fig, ax = dark_fig(6,4)
        ax.plot(sd.index, sd.values, color=ROSE, lw=2.5, marker="o", ms=5)
        ax.fill_between(sd.index, sd.values, alpha=0.15, color=ROSE)
        ax.set_title("Depression Rate vs Combined Stress", fontweight="bold")
        ax.set_xlabel("Stress Index"); ax.set_ylabel("Depression %")
        render_fig(fig)

    with ai2:
        if "Study Satisfaction" in df.columns:
            sat = df.groupby("Study Satisfaction")["Depression"].mean()*100
            fig, ax = dark_fig(6,4)
            ax.plot(sat.index, sat.values, color=AMBER, lw=2.5, marker="^", ms=6)
            ax.fill_between(sat.index, sat.values, alpha=0.15, color=AMBER)
            ax.invert_xaxis()
            ax.set_title("Low Study Satisfaction → High Depression", fontweight="bold")
            ax.set_xlabel("Study Satisfaction (5→1)"); ax.set_ylabel("Depression %")
            render_fig(fig)

    ai3, ai4 = st.columns(2)
    with ai3:
        if "Academic Pressure" in df.columns:
            ap = df.groupby("Academic Pressure")["Work/Study Hours"].mean()
            fig, ax = dark_fig(6,4)
            ax.plot(ap.index, ap.values, color=BLUE, lw=2.5, marker="s", ms=6)
            ax.fill_between(ap.index, ap.values, alpha=0.15, color=BLUE)
            ax.set_title("Academic Pressure → Study Hours", fontweight="bold")
            ax.set_xlabel("Pressure (1-5)"); ax.set_ylabel("Avg Study Hours")
            render_fig(fig)

    with ai4:
        slp_ord = ["Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"]
        df["_slp"] = df["Sleep Duration"].astype(str).str.strip("'")
        slp = df.groupby("_slp")["Depression"].mean()*100
        slp = slp.reindex([s for s in slp_ord if s in slp.index])
        fig, ax = dark_fig(6,4)
        ax.bar(["<5h","5-6h","7-8h",">8h"][:len(slp)], slp.values,
               color=[ROSE,AMBER,TEAL,BLUE][:len(slp)], edgecolor=CARD, alpha=0.85)
        ax.set_title("Sleep Duration vs Depression %", fontweight="bold"); ax.set_ylabel("%")
        render_fig(fig)

    st.markdown("<div class='sh'>⚠️ Risk Accumulation Curve</div>", unsafe_allow_html=True)
    df["_rc"] = (
        (df["Have you ever had suicidal thoughts ?"].astype(str).str.strip("'")=="Yes").astype(int) +
        (pd.to_numeric(df["Financial Stress"],errors="coerce")>=4).astype(int) +
        (df["Sleep Duration"].astype(str).str.strip("'")=="Less than 5 hours").astype(int) +
        (df["Family History of Mental Illness"].astype(str).str.strip("'")=="Yes").astype(int)
    )
    rc = df.groupby("_rc")["Depression"].mean()*100
    fig, ax = dark_fig(12,4)
    ax.plot(rc.index, rc.values, color="#c0392b", lw=3, marker="D", ms=9)
    ax.fill_between(rc.index, rc.values, alpha=0.2, color="#c0392b")
    ax.set_title("How Depression Risk Explodes with More Risk Factors", fontweight="bold")
    ax.set_xlabel("Active Risk Factors"); ax.set_ylabel("Depression Rate (%)")
    render_fig(fig)

    bi1, bi2 = st.columns([1,2])
    with bi1:
        if "City" in df.columns:
            st.markdown("<div class='sh'>🗺️ Top 10 Cities</div>", unsafe_allow_html=True)
            cities = df[df["Depression"]==1]["City"].value_counts().head(10)
            fig, ax = dark_fig(5,5)
            ax.barh(cities.index[::-1], cities.values[::-1],
                    color=[ROSE,AMBER,BLUE,TEAL,PURPLE]*3, edgecolor=CARD, alpha=0.9)
            ax.set_title("Most Depressed by City", fontweight="bold"); ax.set_xlabel("Count")
            render_fig(fig)

    with bi2:
        if "CGPA" in df.columns:
            st.markdown("<div class='sh'>🎓 CGPA vs Depression by Age</div>", unsafe_allow_html=True)
            df["_abin"] = pd.cut(pd.to_numeric(df["Age"],errors="coerce"), bins=range(17,62,2))
            ca = df.groupby(["_abin","Depression"])["CGPA"].mean().unstack(fill_value=np.nan)
            mids = [i.mid for i in ca.index]
            fig, ax = dark_fig(8,5)
            if 0 in ca: ax.plot(mids, ca[0], color=TEAL, lw=2, marker="o", ms=4, label="No Depression")
            if 1 in ca: ax.plot(mids, ca[1], color=ROSE, lw=2, marker="o", ms=4, label="Depressed")
            ax.set_title("Depressed Students — CGPA by Age", fontweight="bold")
            ax.set_xlabel("Age"); ax.set_ylabel("Avg CGPA"); ax.legend(fontsize=9, framealpha=0.2)
            render_fig(fig)


st.markdown(f"""
<div style='text-align:center;padding:14px 0 6px;color:{MUTED};font-size:11px;
            border-top:1px solid {BORDER};margin-top:24px'>
  MindPulse · Streamlit + scikit-learn · Educational use only ·
  Crisis: <b style='color:{ROSE}'>iCall 9152987821</b> ·
  <b style='color:{ROSE}'>Vandrevala 1860-2662-345</b>
</div>
""", unsafe_allow_html=True)