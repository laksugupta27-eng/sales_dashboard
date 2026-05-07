import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SalesIQ — AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
.stApp { background-color: #0e1117; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a1f2e 100%);
    border-right: 1px solid #2a3550;
}
section[data-testid="stSidebar"] * { color: #c9d6e3 !important; }
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #7a8fa6 !important; font-size: 13px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #4fc3f7 !important; font-size: 28px !important; font-weight: 700 !important; }
.page-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d2137 100%);
    border: 1px solid #2a4a7f;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.page-banner h1 { color: #4fc3f7; font-size: 26px; font-weight: 700; margin: 0 0 4px 0; }
.page-banner p { color: #7a9cc0; font-size: 14px; margin: 0; }
.section-header {
    color: #4fc3f7;
    font-size: 16px;
    font-weight: 600;
    border-left: 3px solid #4fc3f7;
    padding-left: 10px;
    margin: 20px 0 12px 0;
}
.chat-wrapper {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 14px;
    padding: 20px;
    max-height: 420px;
    overflow-y: auto;
}
.msg-user {
    background: #1a3050;
    border-radius: 10px 10px 2px 10px;
    padding: 10px 14px;
    margin: 8px 0 8px 60px;
    color: #e0e0e0;
    font-size: 14px;
    border-left: 3px solid #4fc3f7;
}
.msg-ai {
    background: #12263a;
    border-radius: 10px 10px 10px 2px;
    padding: 10px 14px;
    margin: 8px 60px 8px 0;
    color: #c9d6e3;
    font-size: 14px;
    border-left: 3px solid #00bfa5;
}
.msg-label-user { font-size: 11px; color: #4fc3f7; margin-bottom: 4px; font-weight: 600; }
.msg-label-ai   { font-size: 11px; color: #00bfa5; margin-bottom: 4px; font-weight: 600; }
.stSelectbox > div > div { background: #1a1f2e !important; border-color: #2a3550 !important; color: #e0e0e0 !important; }
.stTextInput > div > div { background: #1a1f2e !important; border-color: #2a3550 !important; }
input { color: #e0e0e0 !important; }
.stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7a8fa6; border-radius: 8px; padding: 8px 20px; font-size: 14px; }
.stTabs [aria-selected="true"] { background: #0d3a5c !important; color: #4fc3f7 !important; }
hr { border-color: #2a3550; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #2a3550; border-radius: 3px; }
[data-testid="stDataFrame"] { border: 1px solid #2a3550; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# GROQ API KEY — reads from environment or secrets.toml
# ─────────────────────────────────────────
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = get_api_key()

# ─────────────────────────────────────────
# DATA LOADING — auto-detects local vs cloud
# ─────────────────────────────────────────
def get_data_dir():
    # Check current directory first (for cloud / local same folder)
    if os.path.exists("monthly_actual.csv"):
        return "."
    # Check uploads folder (Claude environment)
    if os.path.exists("/mnt/user-data/uploads/monthly_actual.csv"):
        return "/mnt/user-data/uploads"
    return "."

DATA_DIR = get_data_dir()

@st.cache_data
def load_data():
    monthly     = pd.read_csv(f"{DATA_DIR}/monthly_actual.csv",          parse_dates=["Order Date"])
    forecast    = pd.read_csv(f"{DATA_DIR}/future_forecast.csv",          parse_dates=["Order Date"])
    seg_actual  = pd.read_csv(f"{DATA_DIR}/segment_monthly_actual.csv",   parse_dates=["Order Date"])
    seg_fore    = pd.read_csv(f"{DATA_DIR}/segment_forecast.csv",         parse_dates=["Order Date"])
    reg_actual  = pd.read_csv(f"{DATA_DIR}/region_monthly_actual.csv",    parse_dates=["Order Date"])
    reg_fore    = pd.read_csv(f"{DATA_DIR}/region_forecast.csv",          parse_dates=["Order Date"])
    cat_actual  = pd.read_csv(f"{DATA_DIR}/category_monthly_actual.csv",  parse_dates=["Order Date"])
    cat_fore    = pd.read_csv(f"{DATA_DIR}/category_forecast.csv",        parse_dates=["Order Date"])
    seasonality = pd.read_csv(f"{DATA_DIR}/seasonality_pattern.csv")
    products    = pd.read_csv(f"{DATA_DIR}/product_trend_scores.csv")
    combined    = pd.read_csv(f"{DATA_DIR}/combined_actual_forecast.csv", parse_dates=["Order Date"])
    return (monthly, forecast, seg_actual, seg_fore, reg_actual,
            reg_fore, cat_actual, cat_fore, seasonality, products, combined)

(monthly, forecast, seg_actual, seg_fore, reg_actual,
 reg_fore, cat_actual, cat_fore, seasonality, products, combined) = load_data()

# ─────────────────────────────────────────
# CHART COLORS
# ─────────────────────────────────────────
COLORS = {
    "blue":   "#4fc3f7",
    "teal":   "#00bfa5",
    "amber":  "#ffb74d",
    "coral":  "#ef5350",
    "purple": "#ba68c8",
    "green":  "#66bb6a",
    "grid":   "#1e2a3a",
    "bg":     "#131928",
    "paper":  "#0e1117"
}
SEG_COLORS = [COLORS["blue"],  COLORS["teal"],  COLORS["amber"]]
REG_COLORS = [COLORS["blue"],  COLORS["teal"],  COLORS["amber"], COLORS["coral"]]
CAT_COLORS = [COLORS["blue"],  COLORS["amber"], COLORS["teal"]]

def dark_layout(title="", h=400):
    return dict(
        title=dict(text=title, font=dict(color="#c9d6e3", size=15), x=0.01),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="#c9d6e3", size=12),
        height=h,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"], font=dict(size=12)),
        hovermode="x unified"
    )

# ─────────────────────────────────────────
# GROQ AI CHAT
# ─────────────────────────────────────────
def build_context():
    total_sales  = monthly["Sales"].sum()
    avg_growth   = monthly["Sales_Growth_%"].mean()
    best_month   = monthly.loc[monthly["Sales"].idxmax(), "Month_Name"]
    best_seg     = seg_actual.groupby("Segment")["Sales"].sum().idxmax()
    best_reg     = reg_actual.groupby("Region")["Sales"].sum().idxmax()
    best_cat     = cat_actual.groupby("Category")["Sales"].sum().idxmax()
    top_product  = products.nlargest(1, "Trend_Slope")["Product Name"].values[0]
    forecast_tot = forecast["Predicted_Sales_Seasonally_Adjusted"].sum()
    peak_month   = seasonality.loc[seasonality["Seasonal_Index"].idxmax(), "Month_Name"]
    growing      = len(products[products["Trend_Label"] == "Growing"])
    declining    = len(products[products["Trend_Label"] == "Declining"])

    return f"""You are SalesIQ, an expert AI data analyst for a Superstore Sales Dashboard.
Answer questions clearly, confidently and concisely using the data below.
Always give specific numbers when available. Keep answers to 2-4 sentences.

ACTUAL DATA (2015-2018):
- Total Sales: ${total_sales:,.0f}
- Average Monthly Growth: {avg_growth:.2f}%
- Best Sales Month (highest ever): {best_month}
- Top Segment by revenue: {best_seg}
- Top Region by revenue: {best_reg}
- Top Category by revenue: {best_cat}
- Fastest Growing Product: {top_product}
- Total months of data: {len(monthly)}
- Growing products: {growing}
- Declining products: {declining}

FORECAST (2019 - next 12 months):
- Total Projected Sales: ${forecast_tot:,.0f}
- Peak Season Month: {peak_month}
- Model: Linear Regression + Seasonal Index adjustment
- Confidence band: +/-10% around central forecast

DASHBOARD PAGES:
- Page 1 Actual Overview: KPIs, sales trend, moving averages, YoY growth
- Page 2 Actual Breakdown: by Segment, Region, Category, Top 10 products
- Page 3 Forecast Overview: 12-month projection, confidence bands, seasonal index chart
- Page 4 Forecast Breakdown: segment/region/category forecasts, product bubble chart
- Page 5 AI Chat: this page

Respond professionally. Do not make up data not listed above."""

def ask_gemini(question, history):
    if not GROQ_API_KEY:
        return "⚠️ API key not configured. Please set GROQ_API_KEY in your environment."

    # Build messages for Groq
    messages = [{"role": "system", "content": build_context()}]
    for h in history[-6:]:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["ai"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.4
            },
            timeout=60
        )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"
        else:
            return "Could not get a response. Please try again."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 24px 0;'>
        <div style='font-size:36px; margin-bottom:6px;'>📊</div>
        <div style='font-size:22px; font-weight:700; color:#4fc3f7; letter-spacing:1px;'>SalesIQ</div>
        <div style='font-size:12px; color:#7a8fa6; margin-top:4px;'>AI-Powered Sales Analytics</div>
        <div style='font-size:11px; color:#3a5a70; margin-top:4px;'>Superstore · 2015–2019</div>
    </div>
    <hr style='border-color:#2a3550; margin:0 0 16px 0;'>
    <div style='font-size:11px; color:#3a5a70; margin-bottom:10px; letter-spacing:1px;'>NAVIGATION</div>
    """, unsafe_allow_html=True)

    pages = [
        ("📈", "Actual Overview",    "KPIs · Trend · YoY Growth"),
        ("🔍", "Actual Breakdown",   "Segment · Region · Category"),
        ("🔮", "Forecast Overview",  "Projection · Confidence Bands"),
        ("🧩", "Forecast Breakdown", "Segments · Products · Bubbles"),
        ("🤖", "AI Chat Assistant",  "Ask anything about your data"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = 0

    for i, (icon, name, desc) in enumerate(pages):
        if st.button(f"{icon}  {name}", key=f"nav_{i}", use_container_width=True):
            st.session_state.page = i
        st.markdown(
            f"<div style='font-size:11px;color:#3a5a70;margin:-6px 0 6px 16px;'>{desc}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:#2a3550;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#3a5a70;letter-spacing:1px;margin-bottom:10px;'>QUICK STATS</div>", unsafe_allow_html=True)

    stats = [
        ("💰", "Total Sales",    f"${monthly['Sales'].sum()/1e6:.2f}M"),
        ("📅", "Data Period",    "2015 – 2018"),
        ("🔮", "Forecast Year",  "2019 (12 months)"),
        ("🏷️", "Products",       f"{len(products):,} tracked"),
        ("🚀", "Growing",        f"{len(products[products['Trend_Label']=='Growing'])} products"),
    ]
    for icon, label, val in stats:
        st.markdown(
            f"<div style='font-size:13px;color:#c9d6e3;margin-bottom:8px;'>"
            f"{icon} {label}: <b style='color:#4fc3f7;'>{val}</b></div>",
            unsafe_allow_html=True
        )

    # API key status indicator
    st.markdown("<hr style='border-color:#2a3550;margin:12px 0;'>", unsafe_allow_html=True)
    if GROQ_API_KEY:
        st.markdown("<div style='font-size:12px;color:#00bfa5;'>🟢 AI Chat: Connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px;color:#ef5350;'>🔴 AI Chat: Key not set</div>", unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════
# PAGE 1 — ACTUAL OVERVIEW
# ═══════════════════════════════════════════════════════
if page == 0:
    st.markdown("""
    <div class='page-banner'>
        <h1>📈 Actual Sales Overview</h1>
        <p>Historical performance · January 2015 to December 2018 · 48 months of data</p>
    </div>""", unsafe_allow_html=True)

    col_f, _, _ = st.columns([1, 2, 2])
    with col_f:
        years    = ["All"] + sorted(monthly["Year"].unique().tolist())
        sel_year = st.selectbox("Filter by Year", years, key="p1_year")

    df_m = monthly if sel_year == "All" else monthly[monthly["Year"] == int(sel_year)]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Sales",        f"${df_m['Sales'].sum()/1e3:.1f}K")
    k2.metric("📈 Avg Monthly Growth",  f"{df_m['Sales_Growth_%'].mean():.2f}%")
    k3.metric("🏦 Cumulative Sales",    f"${df_m['Cumulative_Sales'].max()/1e6:.2f}M")
    k4.metric("🚀 Growing Products",    f"{len(products[products['Trend_Label']=='Growing'])}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Sales trend line chart
    st.markdown("<div class='section-header'>Monthly Sales Trend with Moving Averages</div>", unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_m["Order Date"], y=df_m["Sales"],
        name="Actual Sales", mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.07)"
    ))
    fig1.add_trace(go.Scatter(
        x=df_m["Order Date"], y=df_m["Moving_Avg_3M"],
        name="3M Moving Avg", mode="lines",
        line=dict(color=COLORS["amber"], width=2, dash="dot")
    ))
    fig1.add_trace(go.Scatter(
        x=df_m["Order Date"], y=df_m["Moving_Avg_6M"],
        name="6M Moving Avg", mode="lines",
        line=dict(color=COLORS["teal"], width=1.5, dash="dash")
    ))
    fig1.update_layout(**dark_layout("Sales Trend 2015–2018", h=380))
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Year-over-Year Growth % by Month</div>", unsafe_allow_html=True)
        yoy = df_m[df_m["YoY_Growth_%"] != 0]
        fig2 = px.bar(
            yoy, x="YoY_Growth_%", y="Month_Name", color="Year",
            orientation="h", barmode="group",
            color_discrete_sequence=[COLORS["blue"], COLORS["teal"], COLORS["amber"], COLORS["coral"]]
        )
        fig2.update_layout(**dark_layout("YoY Growth %", h=380))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Sales Level Distribution</div>", unsafe_allow_html=True)
        level_counts = df_m["Sales_Level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        fig3 = px.pie(
            level_counts, names="Level", values="Count",
            color_discrete_sequence=[COLORS["blue"], COLORS["coral"]], hole=0.55
        )
        fig3.update_traces(textfont_color="white", textfont_size=13)
        fig3.update_layout(**dark_layout("High vs Low Sales Months", h=380))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Cumulative Sales Growth Over Time</div>", unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_m["Order Date"], y=df_m["Cumulative_Sales"],
        name="Cumulative Sales", mode="lines",
        line=dict(color=COLORS["teal"], width=3),
        fill="tozeroy", fillcolor="rgba(0,191,165,0.07)"
    ))
    fig4.update_layout(**dark_layout("Cumulative Sales Over Time", h=300))
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — ACTUAL BREAKDOWN
# ═══════════════════════════════════════════════════════
elif page == 1:
    st.markdown("""
    <div class='page-banner'>
        <h1>🔍 Actual Sales Breakdown</h1>
        <p>Deep dive into sales performance by Segment · Region · Category · Product</p>
    </div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    years2 = ["All"] + sorted(seg_actual["Order Date"].dt.year.unique().tolist())
    segs   = ["All"] + sorted(seg_actual["Segment"].unique().tolist())
    cats   = ["All"] + sorted(cat_actual["Category"].unique().tolist())
    with f1: sel_y2  = st.selectbox("Year",     years2, key="p2_year")
    with f2: sel_seg = st.selectbox("Segment",  segs,   key="p2_seg")
    with f3: sel_cat = st.selectbox("Category", cats,   key="p2_cat")

    def filt(df, year, col=None, val=None):
        d = df.copy()
        if year != "All": d = d[d["Order Date"].dt.year == int(year)]
        if col and val != "All": d = d[d[col] == val]
        return d

    seg_f = filt(seg_actual, sel_y2, "Segment", sel_seg)
    reg_f = filt(reg_actual, sel_y2)
    cat_f = filt(cat_actual, sel_y2, "Category", sel_cat)

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Sales by Segment</div>", unsafe_allow_html=True)
        fig_seg = go.Figure()
        for i, seg in enumerate(seg_f["Segment"].unique()):
            d = seg_f[seg_f["Segment"] == seg]
            fig_seg.add_trace(go.Scatter(
                x=d["Order Date"], y=d["Sales"], name=seg,
                mode="lines", line=dict(color=SEG_COLORS[i % 3], width=2)
            ))
        fig_seg.update_layout(**dark_layout("Segment Sales Trend", h=340))
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Sales by Region</div>", unsafe_allow_html=True)
        fig_reg = go.Figure()
        for i, reg in enumerate(reg_f["Region"].unique()):
            d = reg_f[reg_f["Region"] == reg]
            fig_reg.add_trace(go.Scatter(
                x=d["Order Date"], y=d["Sales"], name=reg,
                mode="lines", line=dict(color=REG_COLORS[i % 4], width=2)
            ))
        fig_reg.update_layout(**dark_layout("Region Sales Trend", h=340))
        st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("<div class='section-header'>Sales by Category over Time</div>", unsafe_allow_html=True)
    fig_cat = go.Figure()
    for i, cat in enumerate(cat_f["Category"].unique()):
        d = cat_f[cat_f["Category"] == cat]
        fig_cat.add_trace(go.Bar(
            x=d["Order Date"], y=d["Sales"],
            name=cat, marker_color=CAT_COLORS[i % 3]
        ))
    fig_cat.update_layout(**dark_layout("Category Monthly Sales", h=340), barmode="group")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("<div class='section-header'>Top 10 Growing Products by Trend Slope</div>", unsafe_allow_html=True)
    top10 = products[products["Trend_Label"] == "Growing"].nlargest(10, "Trend_Slope")
    fig_prod = px.bar(
        top10, x="Trend_Slope", y="Product Name",
        orientation="h", color="Category",
        color_discrete_sequence=CAT_COLORS, text="Trend_Slope"
    )
    fig_prod.update_traces(texttemplate="%{text:.1f}", textposition="outside", textfont_color="white")
    fig_prod.update_layout(**dark_layout("Top 10 Growing Products", h=420))
    fig_prod.update_yaxes(tickfont=dict(size=11))
    st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown("<div class='section-header'>Segment Summary Table</div>", unsafe_allow_html=True)
    seg_summary = seg_actual.groupby("Segment").agg(
        Total_Sales=("Sales","sum"),
        Avg_Monthly=("Sales","mean"),
        Peak_Sales=("Sales","max")
    ).reset_index().round(2)
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — FORECAST OVERVIEW
# ═══════════════════════════════════════════════════════
elif page == 2:
    st.markdown("""
    <div class='page-banner'>
        <h1>🔮 Forecast Overview</h1>
        <p>12-month sales projection for 2019 · Linear Regression + Seasonal Adjustment + Confidence Bands</p>
    </div>""", unsafe_allow_html=True)

    tot_forecast = forecast["Predicted_Sales_Seasonally_Adjusted"].sum()
    avg_fg       = forecast["Forecast_Growth_%"].mean()
    high_months  = len(forecast[forecast["Forecast_Level"].str.lower().str.strip() == "high"])
    peak_m       = seasonality.loc[seasonality["Seasonal_Index"].idxmax(), "Month_Name"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔮 Total Forecast 2019",  f"${tot_forecast/1e3:.1f}K")
    k2.metric("📊 Avg Forecast Growth",  f"{avg_fg:.2f}%")
    k3.metric("🔥 High Forecast Months", f"{high_months}")
    k4.metric("🌟 Peak Season Month",     peak_m)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Actual + Forecast combined
    st.markdown("<div class='section-header'>Actual vs Forecast — Full Timeline (2015–2019)</div>", unsafe_allow_html=True)
    actual_c   = combined[combined["Is_Forecast"] == "Actual"]
    forecast_c = combined[combined["Is_Forecast"] == "Forecast"]

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(
        x=actual_c["Order Date"], y=actual_c["Sales_Value"],
        name="Actual Sales", mode="lines",
        line=dict(color=COLORS["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.06)"
    ))
    fig_cf.add_trace(go.Scatter(
        x=forecast_c["Order Date"], y=forecast_c["Sales_Value"],
        name="Forecast", mode="lines",
        line=dict(color=COLORS["teal"], width=2.5, dash="dot")
    ))
    fig_cf.add_trace(go.Scatter(
        x=forecast_c["Order Date"], y=forecast_c["Forecast_Upper"],
        name="Upper Band (+10%)", mode="lines",
        line=dict(color=COLORS["amber"], width=1, dash="dash")
    ))
    fig_cf.add_trace(go.Scatter(
        x=forecast_c["Order Date"], y=forecast_c["Forecast_Lower"],
        name="Lower Band (-10%)", mode="lines",
        line=dict(color=COLORS["coral"], width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(0,191,165,0.05)"
    ))
    # vertical line marking forecast start
    forecast_start = actual_c["Order Date"].max()
    fig_cf.add_shape(
        type="line",
        x0=forecast_start, x1=forecast_start,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#7a8fa6", width=1.5, dash="dot")
    )
    fig_cf.add_annotation(
        x=forecast_start, y=1,
        xref="x", yref="paper",
        text="Forecast Start →",
        showarrow=False,
        font=dict(color="#7a8fa6", size=12),
        xanchor="left"
    )
    fig_cf.update_layout(**dark_layout("Actual (2015–2018) + Forecast (2019)", h=420))
    st.plotly_chart(fig_cf, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Seasonal Index by Month</div>", unsafe_allow_html=True)
        bar_colors = [COLORS["teal"] if x >= 1 else COLORS["coral"] for x in seasonality["Seasonal_Index"]]
        fig_si = go.Figure()
        fig_si.add_trace(go.Bar(
            x=seasonality["Month_Name"], y=seasonality["Seasonal_Index"],
            marker_color=bar_colors, name="Seasonal Index",
            text=seasonality["Seasonal_Index"].round(2),
            textposition="outside", textfont=dict(color="white", size=11)
        ))
        fig_si.add_hline(y=1.0, line_dash="dot", line_color="#7a8fa6",
                         annotation_text="Average = 1.0", annotation_font_color="#7a8fa6")
        fig_si.update_layout(**dark_layout("Seasonal Index — Peak vs Slow Months", h=350))
        st.plotly_chart(fig_si, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>2019 Monthly Forecast</div>", unsafe_allow_html=True)
        bar_col2 = [COLORS["teal"] if str(l).lower().strip() == "high" else COLORS["coral"]
                    for l in forecast["Forecast_Level"]]
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Bar(
            x=forecast["Month_Name"],
            y=forecast["Predicted_Sales_Seasonally_Adjusted"],
            marker_color=bar_col2, name="Projected Sales",
            text=forecast["Predicted_Sales_Seasonally_Adjusted"].round(0),
            textposition="outside", textfont=dict(color="white", size=10)
        ))
        fig_fore.update_layout(**dark_layout("2019 Monthly Forecast", h=350))
        st.plotly_chart(fig_fore, use_container_width=True)

    st.markdown("<div class='section-header'>Forecast Detail Table</div>", unsafe_allow_html=True)
    fore_table = forecast[[
        "Month_Name", "Year",
        "Predicted_Sales_Seasonally_Adjusted",
        "Forecast_Upper", "Forecast_Lower",
        "Seasonal_Index", "Forecast_Level"
    ]].copy().round(2)
    fore_table.columns = ["Month","Year","Projected Sales","Upper Bound","Lower Bound","Seasonal Index","Level"]
    st.dataframe(fore_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 4 — FORECAST BREAKDOWN
# ═══════════════════════════════════════════════════════
elif page == 3:
    st.markdown("""
    <div class='page-banner'>
        <h1>🧩 Forecast Breakdown</h1>
        <p>2019 projections by Segment · Region · Category · Product Intelligence Bubble Chart</p>
    </div>""", unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1: sel_sf = st.selectbox("Segment", ["All"] + sorted(seg_fore["Segment"].unique().tolist()), key="p4_seg")
    with f2: sel_cf = st.selectbox("Category", ["All"] + sorted(products["Category"].unique().tolist()), key="p4_cat")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Segment Forecast 2019</div>", unsafe_allow_html=True)
        sf = seg_fore if sel_sf == "All" else seg_fore[seg_fore["Segment"] == sel_sf]
        fig_sf = go.Figure()
        for i, seg in enumerate(sf["Segment"].unique()):
            d = sf[sf["Segment"] == seg]
            fig_sf.add_trace(go.Scatter(
                x=d["Order Date"], y=d["Predicted_Sales"], name=seg,
                mode="lines+markers",
                line=dict(color=SEG_COLORS[i % 3], width=2), marker=dict(size=6)
            ))
        fig_sf.update_layout(**dark_layout("Predicted Sales by Segment", h=340))
        st.plotly_chart(fig_sf, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Region Forecast 2019</div>", unsafe_allow_html=True)
        fig_rf = go.Figure()
        for i, reg in enumerate(reg_fore["Region"].unique()):
            d = reg_fore[reg_fore["Region"] == reg]
            fig_rf.add_trace(go.Scatter(
                x=d["Order Date"], y=d["Predicted_Sales"], name=reg,
                mode="lines+markers",
                line=dict(color=REG_COLORS[i % 4], width=2), marker=dict(size=6)
            ))
        fig_rf.update_layout(**dark_layout("Predicted Sales by Region", h=340))
        st.plotly_chart(fig_rf, use_container_width=True)

    st.markdown("<div class='section-header'>Category Forecast — Monthly Breakdown</div>", unsafe_allow_html=True)
    cf = cat_fore if sel_cf == "All" else cat_fore[cat_fore["Category"] == sel_cf]
    fig_catf = go.Figure()
    for i, cat in enumerate(cf["Category"].unique()):
        d = cf[cf["Category"] == cat]
        fig_catf.add_trace(go.Bar(
            x=d["Order Date"].dt.strftime("%b"), y=d["Predicted_Sales"],
            name=cat, marker_color=CAT_COLORS[i % 3]
        ))
    fig_catf.update_layout(**dark_layout("Category Forecast 2019", h=340), barmode="stack")
    st.plotly_chart(fig_catf, use_container_width=True)

    st.markdown("<div class='section-header'>Product Intelligence — Growth vs Revenue Bubble Chart</div>", unsafe_allow_html=True)
    prod_f = products if sel_cf == "All" else products[products["Category"] == sel_cf]
    fig_bubble = px.scatter(
        prod_f, x="Total_Sales", y="Trend_Slope",
        size="Avg_Monthly_Sales",
        color="Trend_Label",
        color_discrete_map={"Growing": COLORS["teal"], "Declining": COLORS["coral"]},
        hover_name="Product Name",
        hover_data={"Category": True, "Sub-Category": True,
                    "Total_Sales": ":.0f", "Trend_Slope": ":.2f"},
        size_max=30
    )
    fig_bubble.add_hline(y=0, line_dash="dot", line_color="#7a8fa6",
                         annotation_text="Growth Threshold", annotation_font_color="#7a8fa6")
    fig_bubble.update_layout(**dark_layout("Product Growth vs Revenue · Bubble size = Avg Monthly Sales", h=480))
    fig_bubble.update_xaxes(title="Total Sales ($)")
    fig_bubble.update_yaxes(title="Trend Slope (Growth Rate)")
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("<div class='section-header'>Product Rankings Table</div>", unsafe_allow_html=True)
    show_label = st.radio("Show", ["All", "Growing", "Declining"], horizontal=True, key="prod_radio")
    prod_show  = prod_f if show_label == "All" else prod_f[prod_f["Trend_Label"] == show_label]
    prod_show  = prod_show.sort_values("Trend_Slope", ascending=False).head(50)
    st.dataframe(
        prod_show[["Product Name","Category","Sub-Category","Total_Sales","Trend_Slope","Trend_Label","Category_Rank"]].round(2),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════
# PAGE 5 — AI CHAT
# ═══════════════════════════════════════════════════════
elif page == 4:
    st.markdown("""
    <div class='page-banner'>
        <h1>🤖 AI Chat Assistant</h1>
        <p>Ask anything about your sales data · Powered by Groq AI (Llama3)</p>
    </div>""", unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not set. Set GROQ_API_KEY in your environment to enable AI chat.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("<div class='section-header'>Suggested Questions</div>", unsafe_allow_html=True)
    suggestions = [
        "Which segment has the highest sales?",
        "What is the total forecast for 2019?",
        "Which month is the peak season?",
        "Which products are declining?",
        "How was YoY growth in 2017?",
        "Which region performs best overall?"
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                with st.spinner("Thinking..."):
                    reply = ask_gemini(s, st.session_state.chat_history)
                st.session_state.chat_history.append({"user": s, "ai": reply})
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Conversation</div>", unsafe_allow_html=True)

    chat_html = "<div class='chat-wrapper'>"
    if not st.session_state.chat_history:
        chat_html += "<div style='color:#3a5a70;text-align:center;padding:40px;font-size:14px;'>No messages yet. Ask a question below or click a suggestion above.</div>"
    for h in st.session_state.chat_history:
        chat_html += f"<div class='msg-label-user'>You</div><div class='msg-user'>{h['user']}</div>"
        chat_html += f"<div class='msg-label-ai'>SalesIQ AI</div><div class='msg-ai'>{h['ai']}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_input = st.text_input("", placeholder="Ask about sales, forecast, products, regions...",
                                   key="chat_input", label_visibility="collapsed")
    with btn_col:
        send = st.button("Send ➤", use_container_width=True, key="send_btn")

    if send and user_input.strip():
        with st.spinner("SalesIQ is thinking..."):
            reply = ask_gemini(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"user": user_input, "ai": reply})
        st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("📋 What data does the AI know about?"):
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Sales (2015–2018) | ${monthly['Sales'].sum():,.0f} |
        | Top Segment | {seg_actual.groupby('Segment')['Sales'].sum().idxmax()} |
        | Top Region | {reg_actual.groupby('Region')['Sales'].sum().idxmax()} |
        | Top Category | {cat_actual.groupby('Category')['Sales'].sum().idxmax()} |
        | Total Products | {len(products)} |
        | Growing Products | {len(products[products['Trend_Label']=='Growing'])} |
        | 2019 Forecast Total | ${forecast['Predicted_Sales_Seasonally_Adjusted'].sum():,.0f} |
        | Peak Season Month | {seasonality.loc[seasonality['Seasonal_Index'].idxmax(), 'Month_Name']} |
        """)
