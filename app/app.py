"""
Streamlit Application for Chicago Crime Intelligence Dashboard
Main application file for interactive crime data visualization and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import sys
from pathlib import Path

# Add parent directory to path to import eda module
sys.path.append(str(Path(__file__).parent.parent))
import eda


# Configure page
st.set_page_config(
    page_title="Chicago Crime Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS for a professional look
st.markdown("""
<style>
    /* Global layout padding adjustments */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px;
    }
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Metric container styling */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #2D3748;
        padding: 20px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }
    div[data-testid="stMetricLabel"] {
        color: #A0AEC0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 14px;
        margin-bottom: 8px;
    }
    div[data-testid="stMetricValue"] {
        color: #00E6FF;
        font-weight: 700;
        font-size: 36px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(0, 0, 0, 0);
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #00E6FF !important;
        color: #00E6FF !important;
        font-weight: 600 !important;
    }

    /* Styled dataframe table */
    .quality-card {
        background-color: #1A1C24;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 1rem;
    }
    .quality-card h4 {
        color: #FAFAFA;
        margin-bottom: 12px;
    }
    .quality-card p {
        color: #A0AEC0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Set global matplotlib styles for dark theme
plt.style.use('dark_background')
sns.set_style("darkgrid", {"axes.facecolor": "#1A1C24", "grid.color": "#2D3748", "figure.facecolor": "#0E1117"})

def style_plot(fig, ax):
    """Helper function to style matplotlib plots for the dark theme."""
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1A1C24')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2D3748')
    ax.spines['bottom'].set_color('#2D3748')
    ax.tick_params(colors='#A0AEC0', labelsize=10)
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')
    ax.title.set_color('#FAFAFA')


@st.cache_data
def load_data():
    """
    Load the cleaned Chicago Crime dataset with caching.
    
    Returns:
        pd.DataFrame: Cleaned crime dataset.
    """
    try:
        df = eda.load_data()
        return df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()


def apply_filters(df, selected_years, selected_crime_types, selected_districts):
    """
    Apply sidebar filters to the dataset.
    
    Args:
        df (pd.DataFrame): Original dataset.
        selected_years (list): List of selected years.
        selected_crime_types (list): List of selected crime types.
        selected_districts (list): List of selected districts.
    
    Returns:
        pd.DataFrame: Filtered dataset.
    """
    filtered_df = df.copy()
    
    # Apply year filter
    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    
    # Apply crime type filter
    if selected_crime_types:
        filtered_df = filtered_df[filtered_df['Primary Type'].isin(selected_crime_types)]
    
    # Apply district filter
    if selected_districts:
        filtered_df = filtered_df[filtered_df['District'].isin(selected_districts)]
    
    return filtered_df


def display_kpis(df):
    """
    Display Key Performance Indicators (KPIs) at the top of the dashboard.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
    """
    # Calculate KPIs
    total_crimes = len(df)
    
    most_common_crime = df['Primary Type'].mode()[0] if len(df) > 0 else "N/A"
    most_common_crime_count = df['Primary Type'].value_counts().iloc[0] if len(df) > 0 else 0
    
    most_dangerous_district = df['District'].mode()[0] if len(df) > 0 else "N/A"
    most_dangerous_district_count = df['District'].value_counts().iloc[0] if len(df) > 0 else 0
    
    arrest_rate = eda.arrest_rate(df) if len(df) > 0 else 0.0
    
    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Crimes",
            value=f"{total_crimes:,}"
        )
    
    with col2:
        st.metric(
            label="Most Common Crime Type",
            value=most_common_crime,
            delta=f"{most_common_crime_count:,} incidents"
        )
    
    with col3:
        st.metric(
            label="Most Dangerous District",
            value=f"District {most_dangerous_district}",
            delta=f"{most_dangerous_district_count:,} incidents"
        )
    
    with col4:
        st.metric(
            label="Arrest Rate",
            value=f"{arrest_rate:.2f}%"
        )


def display_yearly_trend(df):
    """
    Display yearly crime trend chart.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
    """
    st.subheader("Yearly Crime Trend")
    
    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    yearly_data = eda.yearly_crime_trend(df)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly_data['Year'], yearly_data['Crime_Count'], 
            marker='o', linewidth=3, markersize=8, color='#00E6FF', 
            markerfacecolor='#0E1117', markeredgewidth=2)
    ax.set_title('Crime Trends by Year', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Crimes', fontsize=11)
    ax.grid(True, color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_monthly_trend(df):
    """
    Display monthly crime trend chart.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
    """
    st.subheader("Monthly Crime Trend")
    
    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    monthly_data = eda.monthly_crime_trend(df)
    monthly_data['Date'] = pd.to_datetime(
        monthly_data[['Year', 'Month']].assign(Day=1)
    )
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly_data['Date'], monthly_data['Crime_Count'], 
                    color='#FF4560', alpha=0.15)
    ax.plot(monthly_data['Date'], monthly_data['Crime_Count'], 
            linewidth=2.5, alpha=0.9, color='#FF4560')
    ax.set_title('Monthly Crime Trends', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Number of Crimes', fontsize=11)
    ax.grid(True, color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_top_crime_types(df):
    """
    Display top 10 crime types bar chart.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
    """
    st.subheader("Top 10 Crime Types")
    
    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    top_crimes = eda.top_crime_types(df, top_n=10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_crimes['Primary Type'], top_crimes['Count'], 
            color='#775DD0', alpha=0.85, height=0.6)
    ax.set_title('Top 10 Crime Types', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Crimes', fontsize=11)
    ax.set_ylabel('Crime Type', fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis='x', color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)
    
    # Add value labels on bars
    for i, v in enumerate(top_crimes['Count']):
        ax.text(v + max(top_crimes['Count']) * 0.01, i, 
                f'{v:,}', va='center', fontsize=10, color='#00E6FF', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_crime_by_hour(df):
    """
    Display crime distribution by hour bar chart.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
    """
    st.subheader("Crime Distribution by Hour of Day")
    
    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    hourly_data = eda.crime_by_hour(df)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly_data['Hour'], hourly_data['Crime_Count'], 
           color='#00E396', alpha=0.85, width=0.7)
    ax.set_title('Crime Distribution by Hour of Day', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day (0-23)', fontsize=11)
    ax.set_ylabel('Number of Crimes', fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.grid(True, axis='y', color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_heatmap(df):
    """
    Display crime hotspot heatmap using Folium.
    
    Args:
        df (pd.DataFrame): Filtered dataset with Latitude and Longitude columns.
    """
    st.subheader("Crime Hotspot Map")
    
    if len(df) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    # Filter out rows with invalid coordinates
    map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    if len(map_df) == 0:
        st.warning("No valid location data available for the selected filters.")
        return
    
    # Sample data if too large for performance (limit to 10,000 points)
    if len(map_df) > 10000:
        map_df = map_df.sample(n=10000, random_state=42)
        st.info(f"Displaying 10,000 randomly sampled points from {len(df):,} total records.")
    
    # Create base map centered on Chicago
    chicago_center = [41.8781, -87.6298]
    m = folium.Map(
        location=chicago_center,
        zoom_start=10,
        tiles='cartodbdark_matter'
    )
    
    # Prepare heatmap data
    heat_data = [[row['Latitude'], row['Longitude']] 
                 for idx, row in map_df.iterrows()]
    
    # Add heatmap layer
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Convert map to HTML and embed using st.components.v1.html
    map_html = m.get_root().render()
    st.components.v1.html(map_html, width=1200, height=500)


# ---------------------------------------------------------------------------
# DATA QUALITY & STATISTICS — New Tab Functions
# ---------------------------------------------------------------------------

@st.cache_data
def compute_data_quality_summary(_df):
    """
    Compute high-level data quality metrics for the dataset.
    
    Args:
        _df (pd.DataFrame): Dataset to analyse (prefixed with _ for cache hashing).
    
    Returns:
        dict: Dictionary containing total_rows, total_columns, duplicate_rows,
              total_missing values.
    """
    return {
        "total_rows": len(_df),
        "total_columns": len(_df.columns),
        "duplicate_rows": int(_df.duplicated().sum()),
        "total_missing": int(_df.isnull().sum().sum()),
    }


@st.cache_data
def compute_column_summary(_df):
    """
    Build a per-column quality summary table.
    
    Args:
        _df (pd.DataFrame): Dataset to analyse.
    
    Returns:
        pd.DataFrame: Table with Column, Data Type, Missing Values,
                      Missing %, and Unique Values.
    """
    records = []
    for col in _df.columns:
        missing = int(_df[col].isnull().sum())
        records.append({
            "Column": col,
            "Data Type": str(_df[col].dtype),
            "Missing Values": missing,
            "Missing %": round(missing / len(_df) * 100, 2) if len(_df) > 0 else 0.0,
            "Unique Values": int(_df[col].nunique()),
        })
    return pd.DataFrame(records)


@st.cache_data
def compute_statistics(_df):
    """
    Compute descriptive statistics for all numeric columns.
    
    Args:
        _df (pd.DataFrame): Dataset to analyse.
    
    Returns:
        pd.DataFrame: Table with Mean, Median, Mode, Std Dev, Min, Max per column.
    """
    numeric_df = _df.select_dtypes(include='number')
    if numeric_df.empty:
        return pd.DataFrame()

    records = []
    for col in numeric_df.columns:
        mode_val = numeric_df[col].mode()
        records.append({
            "Column": col,
            "Mean": round(numeric_df[col].mean(), 4),
            "Median": round(numeric_df[col].median(), 4),
            "Mode": round(mode_val.iloc[0], 4) if not mode_val.empty else np.nan,
            "Std Dev": round(numeric_df[col].std(), 4),
            "Min": round(numeric_df[col].min(), 4),
            "Max": round(numeric_df[col].max(), 4),
        })
    return pd.DataFrame(records)


def display_data_quality(df):
    """
    Render the Data Quality section inside the new tab.
    
    Shows high-level KPI cards and a column-wise summary table.
    """
    st.subheader("Data Quality Overview")
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    summary = compute_data_quality_summary(df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Total Rows", value=f"{summary['total_rows']:,}")
    with c2:
        st.metric(label="Total Columns", value=f"{summary['total_columns']}")
    with c3:
        st.metric(label="Duplicate Rows", value=f"{summary['duplicate_rows']:,}")
    with c4:
        st.metric(label="Total Missing Values", value=f"{summary['total_missing']:,}")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Column-wise summary table
    st.markdown("##### Column-wise Quality Summary")
    col_summary = compute_column_summary(df)
    st.dataframe(
        col_summary.style.background_gradient(
            subset=["Missing %"], cmap="YlOrRd", vmin=0, vmax=100
        ).format({"Missing %": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
        height=min(len(col_summary) * 38 + 40, 500),
    )


def display_statistics(df):
    """
    Render descriptive statistics for numeric columns in a styled table.
    """
    st.subheader("Statistical Analysis — Numeric Columns")
    stats = compute_statistics(df)

    if stats.empty:
        st.info("No numeric columns found in the dataset.")
        return

    st.dataframe(
        stats.style.format(
            {c: "{:.4f}" for c in ["Mean", "Median", "Mode", "Std Dev", "Min", "Max"]}
        ).set_properties(**{"text-align": "right"}),
        use_container_width=True,
        hide_index=True,
        height=min(len(stats) * 38 + 40, 400),
    )


def display_missing_chart(df):
    """
    Render a horizontal bar chart of missing values per column.
    """
    st.subheader("Missing Values by Column")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        st.success("No missing values detected in any column.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.5)))
    bars = ax.barh(missing.index, missing.values, color='#FEB019', alpha=0.85, height=0.6)
    ax.set_title('Missing Values per Column', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Missing Count', fontsize=11)
    ax.grid(True, axis='x', color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)

    for bar, val in zip(bars, missing.values):
        ax.text(val + max(missing.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=10, color='#00E6FF', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_correlation(df):
    """
    Render a seaborn heatmap of correlations between numeric columns.
    """
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are required for a correlation heatmap.")
        return

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr.columns) * 0.9),
                                    max(6, len(corr.columns) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor='#2D3748',
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 10, "color": "#FAFAFA"},
    )
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#A0AEC0', labelsize=10)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1A1C24')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_boxplots(df):
    """
    Render box plots for numeric columns to visualise outliers.
    """
    st.subheader("Box Plots — Outlier Detection")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for box plots.")
        return

    selected_col = st.selectbox(
        "Select a numeric column",
        options=numeric_cols,
        key="boxplot_col_select",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        df[selected_col].dropna(),
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='#775DD0', color='#00E6FF', alpha=0.7),
        whiskerprops=dict(color='#A0AEC0'),
        capprops=dict(color='#A0AEC0'),
        medianprops=dict(color='#00E6FF', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='#FF4560', markersize=5, alpha=0.6),
    )
    ax.set_title(f'Box Plot — {selected_col}', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(selected_col, fontsize=11)
    ax.grid(True, axis='y', color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_distribution(df):
    """
    Render histogram + KDE for a selected numeric column.
    """
    st.subheader("Distribution Plot — Histogram + KDE")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for distribution plots.")
        return

    selected_col = st.selectbox(
        "Select a numeric column",
        options=numeric_cols,
        key="dist_col_select",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    data = df[selected_col].dropna()
    ax.hist(data, bins=50, color='#00E396', alpha=0.5, edgecolor='#1A1C24', density=True, label='Histogram')

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde = gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color='#00E6FF', linewidth=2.5, label='KDE')
    except Exception:
        pass  # Skip KDE if scipy is not available

    ax.set_title(f'Distribution — {selected_col}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(selected_col, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(facecolor='#1A1C24', edgecolor='#2D3748', labelcolor='#A0AEC0')
    ax.grid(True, axis='y', color='#2D3748', alpha=0.5, linestyle='--')
    style_plot(fig, ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Main function to run the Streamlit application.
    """
    # Stylish Title
    st.markdown("""
        <div style='margin-bottom: 1.5rem; padding-top: 0.5rem;'>
            <h1 style='display: flex; align-items: center; justify-content: flex-start; gap: 12px; text-align: left; font-size: 38px; font-weight: 700; margin-bottom: 8px; color: #FAFAFA;'>
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#00E6FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
                <span><span style='color: #00E6FF;'>Chicago</span> Crime Intelligence</span>
            </h1>
            <p style='font-size: 15px; color: #A0AEC0; margin-top: 0; padding-left: 46px;'>
                Interactive dashboard for analyzing crime patterns, hotspots, and trends across Chicago.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters styling
    st.sidebar.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem; margin-top: 0.5rem;'>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00E6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
            </svg>
            <h2 style='font-size: 18px; color: #FAFAFA; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Filters</h2>
            <p style='color: #A0AEC0; font-size: 13px; margin-top: 4px; margin-bottom: 0;'>Refine dashboard data</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Get unique values for filters
    available_years = sorted(df['Year'].unique())
    available_crime_types = sorted(df['Primary Type'].unique())
    available_districts = sorted(df['District'].dropna().unique())
    
    # Year selector
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=available_years,
        default=available_years,
        help="Select one or more years to filter the data"
    )
    
    # Crime Type selector
    selected_crime_types = st.sidebar.multiselect(
        "Select Crime Types",
        options=available_crime_types,
        default=[],
        help="Select one or more crime types to filter the data"
    )
    
    # District selector
    selected_districts = st.sidebar.multiselect(
        "Select Districts",
        options=available_districts,
        default=[],
        help="Select one or more districts to filter the data"
    )
    
    # Apply filters
    filtered_df = apply_filters(df, selected_years, selected_crime_types, selected_districts)
    
    # Display record count prominently
    st.sidebar.divider()
    st.sidebar.markdown("""
        <div style='background-color: #1A1C24; padding: 15px; border-radius: 8px; border: 1px solid #2D3748; text-align: center;'>
            <p style='color: #A0AEC0; font-size: 14px; text-transform: uppercase; margin-bottom: 5px;'>Filtered Records</p>
            <h3 style='color: #00E6FF; font-size: 24px; margin: 0;'>{:,}</h3>
        </div>
    """.format(len(filtered_df)), unsafe_allow_html=True)
    
    # Display KPIs
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    display_kpis(filtered_df)
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Visual sections embedded in professional tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview & Trends", 
        "Demographics & Time", 
        "Geospatial Analysis",
        "Data Quality & Statistics",
    ])
    
    with tab1:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            display_yearly_trend(filtered_df)
        with col2:
            display_monthly_trend(filtered_df)
            
    with tab2:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            display_top_crime_types(filtered_df)
        with col4:
            display_crime_by_hour(filtered_df)
            
    with tab3:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        display_heatmap(filtered_df)

    with tab4:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # --- Section 1: Data Quality ---
        display_data_quality(filtered_df)

        st.divider()

        # --- Section 2: Statistical Analysis ---
        display_statistics(filtered_df)

        st.divider()

        # --- Section 3: Visualizations ---
        st.subheader("Advanced Visualizations")
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        viz_left, viz_right = st.columns(2)

        with viz_left:
            display_missing_chart(filtered_df)
        with viz_right:
            display_correlation(filtered_df)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        box_col, dist_col = st.columns(2)

        with box_col:
            display_boxplots(filtered_df)
        with dist_col:
            display_distribution(filtered_df)


if __name__ == "__main__":
    main()
