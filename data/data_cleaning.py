"""
Data Cleaning Script for Crime Intelligence Dashboard
Processes Chicago Crime dataset (Crimes - 2001 to Present)
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_crime_data():
    """
    Main function to clean the Chicago Crime dataset.
    
    Steps performed:
    1. Load dataset with specific columns
    2. Convert Date column to datetime format
    3. Filter to last 5 years of data
    4. Remove rows with null values in critical columns
    5. Create new feature columns from Date
    6. Reset index and save cleaned dataset
    """
    
    # Step 1: Load dataset with specific columns only
    print("Loading dataset...")
    input_file = "data/chicago_crime.csv"
    
    # Define columns to load
    columns_to_load = [
        'Date',
        'Primary Type',
        'Arrest',
        'District',
        'Latitude',
        'Longitude'
    ]
    
    try:
        df = pd.read_csv(input_file, usecols=columns_to_load)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Required column not found in dataset: {e}")
        return None
    
    # Store original shape for reporting
    original_shape = df.shape
    
    # Step 2: Convert Date column to pandas datetime format
    print("\nConverting Date column to datetime format...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Check for any conversion errors
    null_dates = df['Date'].isna().sum()
    if null_dates > 0:
        print(f"Warning: {null_dates} rows have invalid dates and will be removed.")
    
    # Step 3: Filter dataset to keep only last 5 years of data
    print("\nFiltering to last 5 years of data...")
    
    # Get the maximum date in the dataset
    max_date = df['Date'].max()
    
    # Calculate the date 5 years ago from the maximum date
    five_years_ago = max_date - pd.DateOffset(years=5)
    
    # Filter rows where Date is within the last 5 years
    df = df[df['Date'] >= five_years_ago].copy()
    
    print(f"Data filtered. Records after filtering: {df.shape[0]}")
    
    # Step 4: Remove rows where Latitude, Longitude, or District is null
    print("\nRemoving rows with null values in critical columns...")
    
    # Count rows before removal
    rows_before_removal = len(df)
    
    # Remove rows where any of these columns is null
    df = df.dropna(subset=['Latitude', 'Longitude', 'District']).copy()
    
    rows_removed = rows_before_removal - len(df)
    print(f"Removed {rows_removed} rows with null values.")
    
    # Step 5: Create new feature columns from Date
    print("\nCreating new feature columns from Date...")
    
    # Extract Year from Date
    df['Year'] = df['Date'].dt.year
    
    # Extract Month from Date (1-12)
    df['Month'] = df['Date'].dt.month
    
    # Extract Hour from Date (0-23)
    df['Hour'] = df['Date'].dt.hour
    
    # Extract Day of Week from Date (Monday=0, Sunday=6)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    print("Feature columns created: Year, Month, Hour, DayOfWeek")
    
    # Step 6: Reset index after cleaning
    print("\nResetting index...")
    df = df.reset_index(drop=True)
    
    # Step 7: Save cleaned dataset
    print("\nSaving cleaned dataset...")
    output_file = "data/cleaned_chicago_crime.csv"
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    # Step 8: Print summary statistics
    print("\n" + "="*60)
    print("DATA CLEANING SUMMARY")
    print("="*60)
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"\nFirst 5 rows of cleaned dataset:")
    print(df.head())
    print(f"\nList of columns:")
    print(df.columns.tolist())
    print("="*60)
    
    return df


if __name__ == "__main__":
    """
    Execute the data cleaning script when run directly.
    """
    print("Starting Crime Intelligence Dashboard - Data Cleaning")
    print("="*60)
    
    cleaned_data = clean_crime_data()
    
    if cleaned_data is not None:
        print("\nData cleaning completed successfully!")
    else:
        print("\nData cleaning failed. Please check the error messages above.")
