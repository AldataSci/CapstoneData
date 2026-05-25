import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Subway Crime Analysis",
    page_icon="🚇",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background-color: #0d0d0d; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .kpi-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .kpi-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.4rem;
  }
  .kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f0f0;
  }
  .kpi-sub {
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 0.3rem;
  }

  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #e85d3a;
    margin: 2rem 0 1rem;
    border-bottom: 1px solid #2e2e2e;
    padding-bottom: 0.5rem;
  }

  .stSelectbox label, .stRadio label, .stMultiSelect label {
    color: #aaa !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }

  .cluster-legend {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 1.5rem;
  }
  .cluster-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: #ccc;
  }
  .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
URL_2019 = "https://raw.githubusercontent.com/AldataSci/CapstoneData/refs/heads/main/clean_cluster_2019.csv"
URL_2022 = "https://raw.githubusercontent.com/AldataSci/CapstoneData/refs/heads/main/clean_cluster.csv"

@st.cache_data(show_spinner="Loading data...")
def load_and_process(url, ridership_col):
    df = pd.read_csv(url, low_memory=False)

    cols = ['Matched_Station', 'complaint_count', ridership_col,
            'latitude', 'longitude', 'boro_nm', 'ofns_desc', 'law_cat_cd', 'month']
    df = df[cols].copy()
    df.columns = ['station', 'complaint_count', 'ridership',
                  'latitude', 'longitude', 'borough', 'offense', 'severity', 'month']

    df['complaint_count'] = pd.to_numeric(df['complaint_count'], errors='coerce')
    df['ridership']       = pd.to_numeric(df['ridership'],       errors='coerce')
    df['latitude']        = pd.to_numeric(df['latitude'],        errors='coerce')
    df['longitude']       = pd.to_numeric(df['longitude'],       errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'ridership', 'complaint_count'])

    # Aggregate to station level
    station = df.groupby('station').agg(
        complaint_count=('complaint_count', 'first'),
        ridership=('ridership', 'first'),
        latitude=('latitude', 'median'),
        longitude=('longitude', 'median'),
        borough=('borough', 'first'),
        top_offense=('offense', lambda x: x.value_counts().idxmax()),
        felony_pct=('severity', lambda x: (x == 'FELONY').mean() * 100),
    ).reset_index()

    station = station[station['ridership'] > 0]
    station['crimes_per_million'] = (
        station['complaint_count'] / (station['ridership'] / 1_000_000)
    ).round(2)

    # K-Means clustering (4 clusters, matching original notebook)
    features = station[['complaint_count', 'crimes_per_million']].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    station['danger_cluster'] = km.fit_predict(scaled)

    # Re-label clusters so 0=safest, 3=most dangerous by crimes_per_million mean
    cluster_means = station.groupby('danger_cluster')['crimes_per_million'].mean().sort_values()
    remap = {old: new for new, old in enumerate(cluster_means.index)}
    station['danger_cluster'] = station['danger_cluster'].map(remap)

    return station, df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚇 NYC Subway\nCrime Analysis")
    st.markdown("---")

    year = st.radio("Year", ["2019 (Pre-Pandemic)", "2022 (Post-Pandemic)"])
    selected_year = "2019" if "2019" in year else "2022"

    st.markdown("---")
    st.markdown("**Filters**")

    borough_opts = ["All", "MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]
    borough = st.selectbox("Borough", borough_opts)

    severity_opts = ["All", "FELONY", "MISDEMEANOR", "VIOLATION"]
    severity = st.selectbox("Severity", severity_opts)

    st.markdown("---")
    st.caption("Data: NYPD Complaint Data + MTA Ridership\nCUNY Capstone · Al Haque · 2024")

# ── Load data ─────────────────────────────────────────────────────────────────
if selected_year == "2019":
    station_df, raw_df = load_and_process(URL_2019, "2019")
else:
    station_df, raw_df = load_and_process(URL_2022, "2022")

# Apply borough filter
filtered_stations = station_df.copy()
if borough != "All":
    filtered_stations = filtered_stations[filtered_stations['borough'] == borough]

# Apply severity filter to raw for offense breakdown
filtered_raw = raw_df.copy()
if borough != "All":
    filtered_raw = filtered_raw[filtered_raw['borough'] == borough]
if severity != "All":
    filtered_raw = filtered_raw[filtered_raw['severity'] == severity]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-family:Space Mono,monospace; font-size:1.6rem; color:#f0f0f0; margin-bottom:0.2rem;'>
NYC Subway Crime & Ridership
</h1>
<p style='color:#888; font-size:0.9rem; margin-top:0;'>
{selected_year} · {borough if borough != "All" else "All Boroughs"} · 
{len(filtered_stations):,} stations analyzed
</p>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
total_incidents  = int(filtered_stations['complaint_count'].sum())
most_dangerous   = filtered_stations.loc[filtered_stations['crimes_per_million'].idxmax(), 'station']
avg_cpm          = filtered_stations['crimes_per_million'].median()
top_offense      = filtered_raw['offense'].value_counts().idxmax() if len(filtered_raw) > 0 else "N/A"
high_danger_ct   = (filtered_stations['danger_cluster'] == 3).sum()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Total Incidents</div>
      <div class="kpi-value">{total_incidents:,}</div>
      <div class="kpi-sub">complaint records</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Most Dangerous Station</div>
      <div class="kpi-value" style="font-size:1.1rem; color:#e85d3a;">{most_dangerous}</div>
      <div class="kpi-sub">by crimes per million riders</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Median Crime Rate</div>
      <div class="kpi-value">{avg_cpm:.1f}</div>
      <div class="kpi-sub">crimes per million riders</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">High-Danger Stations</div>
      <div class="kpi-value" style="color:#e85d3a;">{high_danger_ct}</div>
      <div class="kpi-sub">cluster 3 (most dangerous)</div>
    </div>""", unsafe_allow_html=True)

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Station Danger Map</div>', unsafe_allow_html=True)

st.markdown("""
<div class="cluster-legend">
  <span class="cluster-badge"><span class="dot" style="background:#1a9850"></span> Cluster 0 — Lowest Danger</span>
  <span class="cluster-badge"><span class="dot" style="background:#fee08b"></span> Cluster 1 — Low-Medium</span>
  <span class="cluster-badge"><span class="dot" style="background:#f46d43"></span> Cluster 2 — Medium-High</span>
  <span class="cluster-badge"><span class="dot" style="background:#d73027"></span> Cluster 3 — Highest Danger</span>
</div>
""", unsafe_allow_html=True)

color_map = {0: "#1a9850", 1: "#fee08b", 2: "#f46d43", 3: "#d73027"}
filtered_stations['color'] = filtered_stations['danger_cluster'].map(color_map)
filtered_stations['cluster_label'] = filtered_stations['danger_cluster'].map({
    0: "Cluster 0 — Safest",
    1: "Cluster 1 — Low-Medium",
    2: "Cluster 2 — Medium-High",
    3: "Cluster 3 — Most Dangerous"
})

fig_map = px.scatter_mapbox(
    filtered_stations,
    lat="latitude", lon="longitude",
    color="cluster_label",
    color_discrete_map={
        "Cluster 0 — Safest":         "#1a9850",
        "Cluster 1 — Low-Medium":     "#fee08b",
        "Cluster 2 — Medium-High":    "#f46d43",
        "Cluster 3 — Most Dangerous": "#d73027",
    },
    size="complaint_count",
    size_max=22,
    hover_name="station",
    hover_data={
        "complaint_count":    True,
        "crimes_per_million": True,
        "borough":            True,
        "cluster_label":      False,
        "latitude":           False,
        "longitude":          False,
    },
    labels={
        "complaint_count":    "Total Complaints",
        "crimes_per_million": "Crimes / Million Riders",
        "borough":            "Borough",
    },
    mapbox_style="carto-darkmatter",
    zoom=10,
    center={"lat": 40.73, "lon": -73.95},
    height=520,
)
fig_map.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="#0d0d0d",
    legend=dict(
        bgcolor="#1a1a1a",
        bordercolor="#2e2e2e",
        borderwidth=1,
        font=dict(color="#ccc", size=11),
    )
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Two-column charts ─────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">Top 10 Most Dangerous Stations</div>', unsafe_allow_html=True)
    top10 = (
        filtered_stations
        .nlargest(10, 'crimes_per_million')[['station', 'crimes_per_million', 'borough', 'danger_cluster']]
        .sort_values('crimes_per_million')
    )
    top10['bar_color'] = top10['danger_cluster'].map(color_map)

    fig_bar = go.Figure(go.Bar(
        x=top10['crimes_per_million'],
        y=top10['station'],
        orientation='h',
        marker_color=top10['bar_color'],
        text=top10['crimes_per_million'].round(1),
        textposition='outside',
        textfont=dict(color='#ccc', size=10),
        hovertemplate='<b>%{y}</b><br>%{x:.1f} crimes/million riders<extra></extra>',
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(color="#ccc"),
        xaxis=dict(gridcolor="#2e2e2e", title="Crimes per Million Riders"),
        yaxis=dict(gridcolor="#2e2e2e"),
        margin=dict(l=10, r=40, t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Crime Type Breakdown</div>', unsafe_allow_html=True)
    offense_counts = (
        filtered_raw['offense']
        .value_counts()
        .head(10)
        .reset_index()
    )
    offense_counts.columns = ['offense', 'count']

    fig_offense = go.Figure(go.Bar(
        x=offense_counts['count'],
        y=offense_counts['offense'],
        orientation='h',
        marker_color='#e85d3a',
        marker_opacity=0.85,
        hovertemplate='<b>%{y}</b><br>%{x:,} incidents<extra></extra>',
    ))
    fig_offense.update_layout(
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(color="#ccc"),
        xaxis=dict(gridcolor="#2e2e2e", title="Incident Count"),
        yaxis=dict(gridcolor="#2e2e2e"),
        margin=dict(l=10, r=20, t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig_offense, use_container_width=True)

# ── Borough breakdown ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Crime Rate by Borough</div>', unsafe_allow_html=True)

boro_summary = (
    station_df.groupby('borough')
    .agg(
        avg_cpm=('crimes_per_million', 'mean'),
        total_complaints=('complaint_count', 'sum'),
        station_count=('station', 'count'),
    )
    .reset_index()
    .sort_values('avg_cpm', ascending=False)
)

fig_boro = go.Figure(go.Bar(
    x=boro_summary['borough'],
    y=boro_summary['avg_cpm'],
    marker_color=['#d73027', '#f46d43', '#fee08b', '#91cf60', '#1a9850'][:len(boro_summary)],
    text=boro_summary['avg_cpm'].round(1),
    textposition='outside',
    textfont=dict(color='#ccc'),
    hovertemplate='<b>%{x}</b><br>Avg %{y:.1f} crimes/million riders<extra></extra>',
))
fig_boro.update_layout(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="#ccc"),
    xaxis=dict(gridcolor="#2e2e2e"),
    yaxis=dict(gridcolor="#2e2e2e", title="Avg Crimes per Million Riders"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=300,
)
st.plotly_chart(fig_boro, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("📋 View Station-Level Data"):
    display_df = filtered_stations[[
        'station', 'borough', 'complaint_count',
        'ridership', 'crimes_per_million', 'cluster_label', 'top_offense'
    ]].sort_values('crimes_per_million', ascending=False).reset_index(drop=True)
    display_df.columns = [
        'Station', 'Borough', 'Total Complaints',
        'Annual Ridership', 'Crimes/Million Riders', 'Danger Cluster', 'Top Offense'
    ]
    st.dataframe(display_df, use_container_width=True, height=300)
