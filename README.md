# NYC Subway Crime & Ridership Analysis (2019 vs 2022)

**Which NYC subway stations are the most dangerous — and did the pandemic make it worse?**

This project analyzes 8M+ NYPD complaint records and MTA ridership data to identify crime hotspots across 400+ subway stations, comparing pre-pandemic (2019) and post-pandemic (2022) patterns using K-Means clustering.

🚇 **[Live App →]([https://your-app-link.streamlit.app](https://capstonedata-fbvy2dwbda5hytdqqqwye9.streamlit.app/)** *(App Deployed)*

---

## The Question

Raw crime counts alone are misleading — a busy station like Times Square will always have more incidents than a quiet outer-borough stop. This project normalizes crime by ridership to find which stations are *truly* dangerous relative to how many people use them.

---

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| NYPD Complaint Data | [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i) | 8M+ records (2006–2024) |
| MTA Subway Ridership | [MTA Open Data](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-2022/wujg-7c2s) | Annual ridership by station |

---

## Methods

**1. Data Cleaning** (`data_cleaning.ipynb`)
- Filtered NYPD complaints to subway jurisdiction only
- Matched complaint records to MTA station names via fuzzy string matching
- Merged crime counts with annual ridership figures for 2019 and 2022

**2. Feature Engineering**
- Computed `crimes_per_million_riders` per station — a normalized danger score that accounts for station volume
- Aggregated to station level: total complaints, average lat/lon, borough, top crime type

**3. K-Means Clustering** (`kmeans_clustering.ipynb`)
- Scaled features with `StandardScaler`
- Applied K-Means with 4 clusters (validated with silhouette score)
- Cluster 0 = lowest danger, Cluster 3 = highest danger

---

## Key Findings

- **Atlantic Av and Broad Channel** had the highest crime-per-rider ratio in 2019, making them outlier danger stations despite not being the busiest stops
- Post-pandemic (2022), crime-per-rider ratios shifted significantly as ridership dropped but incidents did not fall proportionally
- Felony assaults and grand larceny accounted for the majority of serious incidents across all clusters
- Outer-borough stations in the Bronx and parts of Brooklyn consistently appeared in the high-danger cluster

---

## Repository Structure

```
├── data_cleaning.ipynb        # NYPD + MTA data cleaning and merging
├── kmeans_clustering.ipynb    # K-Means clustering and visualization
├── clean_cluster.csv          # Cleaned incident-level data (2022)
├── clean_cluster_2019.csv     # Cleaned incident-level data (2019)
└── README.md
```

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `NumPy` · `Matplotlib` · `Streamlit` · `Folium`

---

## Author

**Al Haque** — M.Sc. Data Science, CUNY School of Professional Studies (2024)

[LinkedIn](https://www.linkedin.com/in/al-haque-a97592233/) · [GitHub](https://github.com/AldataSci)



Youtube Video: https://www.youtube.com/watch?v=lHgX__rMGmE&ab_channel=AlHaque (youtube Link to my Capstone Project presentation..)
