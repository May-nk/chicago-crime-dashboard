"""
Exploratory Data Analysis Module for Crime Intelligence Dashboard
Contains reusable analytical and visualization functions for crime data analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data():
    """
    Load the cleaned Chicago Crime dataset.
    
    Returns:
        pd.DataFrame: Cleaned crime dataset with columns:
            - Date, Primary Type, Arrest, District, Latitude, Longitude
            - Year, Month, Hour, DayOfWeek (derived features)
    
    Raises:
        FileNotFoundError: If the cleaned dataset file is not found.
    """
    data_path = Path("data/cleaned_chicago_crime.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. "
            "Please run data_cleaning.py first."
        )
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    return df


def yearly_crime_trend(df):
    """
    Calculate crime count per year.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Year' column.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Year', 'Crime_Count'] 
                      sorted by Year.
    """
    yearly_counts = df.groupby('Year').size().reset_index(name='Crime_Count')
    yearly_counts = yearly_counts.sort_values('Year')
    return yearly_counts


def monthly_crime_trend(df):
    """
    Calculate monthly aggregated crime counts grouped by Year and Month.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Year' and 'Month' columns.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Year', 'Month', 'Crime_Count']
                      sorted by Year and Month.
    """
    monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Crime_Count')
    monthly_counts = monthly_counts.sort_values(['Year', 'Month'])
    return monthly_counts


def top_crime_types(df, top_n=10):
    """
    Return top N most frequent crime categories.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Primary Type' column.
        top_n (int): Number of top crime types to return. Default is 10.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Primary Type', 'Count']
                      sorted by Count in descending order.
    """
    crime_counts = df['Primary Type'].value_counts().head(top_n).reset_index()
    crime_counts.columns = ['Primary Type', 'Count']
    return crime_counts


def top_districts(df, top_n=10):
    """
    Return top N districts with highest crime count.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'District' column.
        top_n (int): Number of top districts to return. Default is 10.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['District', 'Crime_Count']
                      sorted by Crime_Count in descending order.
    """
    district_counts = df['District'].value_counts().head(top_n).reset_index()
    district_counts.columns = ['District', 'Crime_Count']
    return district_counts


def crime_by_hour(df):
    """
    Calculate distribution of crimes by hour of day.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Hour' column.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Hour', 'Crime_Count']
                      sorted by Hour (0-23).
    """
    hourly_counts = df.groupby('Hour').size().reset_index(name='Crime_Count')
    hourly_counts = hourly_counts.sort_values('Hour')
    return hourly_counts


def arrest_rate(df):
    """
    Calculate percentage of crimes resulting in arrest.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Arrest' column.
    
    Returns:
        float: Percentage of crimes resulting in arrest (0-100).
    """
    if 'Arrest' not in df.columns:
        raise ValueError("'Arrest' column not found in dataset.")
    
    # Convert Arrest column to boolean if it's not already
    arrest_series = df['Arrest']
    if arrest_series.dtype == 'object':
        # Handle string values like 'True', 'False', 'Y', 'N', etc.
        arrest_series = arrest_series.astype(str).str.upper()
        arrest_bool = arrest_series.isin(['TRUE', 'T', 'YES', 'Y', '1'])
    else:
        arrest_bool = arrest_series.astype(bool)
    
    arrest_percentage = (arrest_bool.sum() / len(df)) * 100
    return round(arrest_percentage, 2)


def plot_yearly_trend(df):
    """
    Generate a line plot showing crime trends by year.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Year' column.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    yearly_data = yearly_crime_trend(df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_data['Year'], yearly_data['Crime_Count'], 
            marker='o', linewidth=2, markersize=8)
    ax.set_title('Crime Trends by Year', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Crimes', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    return fig


def plot_monthly_trend(df):
    """
    Generate a line plot showing monthly crime trends across years.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Year' and 'Month' columns.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    monthly_data = monthly_crime_trend(df)
    
    # Create a date column for better x-axis representation
    monthly_data['Date'] = pd.to_datetime(
        monthly_data[['Year', 'Month']].assign(Day=1)
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_data['Date'], monthly_data['Crime_Count'], 
            linewidth=2, alpha=0.8)
    ax.set_title('Monthly Crime Trends', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Crimes', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_top_crime_types(df, top_n=10):
    """
    Generate a horizontal bar plot showing top N crime types.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Primary Type' column.
        top_n (int): Number of top crime types to display. Default is 10.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    top_crimes = top_crime_types(df, top_n=top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_crimes['Primary Type'], top_crimes['Count'], 
            color='steelblue', alpha=0.8)
    ax.set_title(f'Top {top_n} Crime Types', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Crimes', fontsize=12)
    ax.set_ylabel('Crime Type', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    
    # Invert y-axis to show highest count at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, v in enumerate(top_crimes['Count']):
        ax.text(v + max(top_crimes['Count']) * 0.01, i, 
                f'{v:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_crime_by_hour(df):
    """
    Generate a bar plot showing distribution of crimes by hour of day.
    
    Args:
        df (pd.DataFrame): Crime dataset with 'Hour' column.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    hourly_data = crime_by_hour(df)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(hourly_data['Hour'], hourly_data['Crime_Count'], 
           color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_title('Crime Distribution by Hour of Day', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day (0-23)', fontsize=12)
    ax.set_ylabel('Number of Crimes', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    """
    Test block to verify functions work correctly.
    """
    print("Loading cleaned dataset...")
    df = load_data()
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    print("="*60)
    print("Testing Analysis Functions")
    print("="*60)
    
    print("\n1. Yearly Crime Trend (first 5 rows):")
    print(yearly_crime_trend(df).head())
    
    print("\n2. Top Crime Types:")
    print(top_crime_types(df))
    
    print("\n3. Top Districts:")
    print(top_districts(df))
    
    print("\n4. Crime by Hour (first 5 rows):")
    print(crime_by_hour(df).head())
    
    print("\n5. Arrest Rate:")
    arrest_pct = arrest_rate(df)
    print(f"Arrest Rate: {arrest_pct}%")
    
    print("\n6. Monthly Crime Trend (first 5 rows):")
    print(monthly_crime_trend(df).head())
    
    print("\n" + "="*60)
    print("All functions executed successfully!")
    print("="*60)
    
    # Uncomment below to generate and save plots
    # print("\nGenerating visualizations...")
    # plot_yearly_trend(df)
    # plt.savefig('yearly_trend.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # 
    # plot_monthly_trend(df)
    # plt.savefig('monthly_trend.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # 
    # plot_top_crime_types(df)
    # plt.savefig('top_crime_types.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # 
    # plot_crime_by_hour(df)
    # plt.savefig('crime_by_hour.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # 
    # print("Visualizations saved successfully!")
