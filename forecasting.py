"""
Forecasting Module for Crime Intelligence Dashboard
Provides time-series forecasting capabilities for monthly crime count prediction.
Uses simple Linear Regression approach for forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path


def prepare_monthly_data(df):
    """
    Prepare monthly aggregated data for time-series forecasting.
    
    Converts the dataset to monthly aggregates with datetime and count columns.
    
    Args:
        df (pd.DataFrame): Cleaned crime dataset with 'Date' column.
                          Should also have 'Year' and 'Month' columns if available.
    
    Returns:
        pd.DataFrame: DataFrame with two columns:
            - ds: datetime column (first day of each month)
            - y: total monthly crime count
    """
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Convert Date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Extract Year and Month if not already present
    if 'Year' not in df_copy.columns or 'Month' not in df_copy.columns:
        df_copy['Year'] = df_copy['Date'].dt.year
        df_copy['Month'] = df_copy['Date'].dt.month
    
    # Aggregate data by Year and Month
    monthly_agg = df_copy.groupby(['Year', 'Month']).size().reset_index(name='y')
    
    # Create datetime column (first day of each month)
    monthly_agg['ds'] = pd.to_datetime(
        monthly_agg[['Year', 'Month']].assign(Day=1)
    )
    
    # Select only ds and y columns, sort by date
    monthly_df = monthly_agg[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    return monthly_df


def train_test_split_time_series(df, test_size=6):
    """
    Split time-series data into training and testing sets.
    
    Uses the last N months as test data, keeping the rest for training.
    Maintains temporal order (no random shuffling).
    
    Args:
        df (pd.DataFrame): Monthly aggregated data with 'ds' and 'y' columns.
        test_size (int): Number of months to use as test data. Default is 6.
    
    Returns:
        tuple: (train_df, test_df) - Training and testing DataFrames.
    """
    # Ensure data is sorted by date
    df_sorted = df.sort_values('ds').reset_index(drop=True)
    
    # Split: last test_size months go to test, rest to train
    split_index = len(df_sorted) - test_size
    
    if split_index <= 0:
        raise ValueError(
            f"Dataset has {len(df_sorted)} months, but test_size is {test_size}. "
            "Need more data for splitting."
        )
    
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    
    return train_df, test_df


def train_forecast_model(train_df):
    """
    Train a Linear Regression model for time-series forecasting.
    
    Converts datetime to ordinal values for model training.
    
    Args:
        train_df (pd.DataFrame): Training data with 'ds' (datetime) and 'y' (target) columns.
    
    Returns:
        sklearn.linear_model.LinearRegression: Trained model.
    """
    # Convert datetime to ordinal (numeric) values
    # Ordinal represents the number of days since a reference date
    X_train = train_df['ds'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_train = train_df['y'].values
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model


def forecast_future(model, df, months=6):
    """
    Generate forecasts for future months.
    
    Predicts crime counts for the next N months based on the trained model.
    
    Args:
        model (sklearn.linear_model.LinearRegression): Trained forecasting model.
        df (pd.DataFrame): Historical monthly data with 'ds' column.
        months (int): Number of future months to forecast. Default is 6.
    
    Returns:
        pd.DataFrame: Forecast DataFrame with columns:
            - ds: datetime (first day of each forecasted month)
            - y_pred: predicted crime count
    """
    # Get the last date in the dataset
    last_date = df['ds'].max()
    
    # Generate next N months datetime values
    # Start from the month after the last date
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=months,
        freq='MS'  # Month Start frequency
    )
    
    # Convert forecast dates to ordinal (same format as training)
    X_forecast = forecast_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    
    # Predict crime counts
    y_pred = model.predict(X_forecast)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'y_pred': y_pred
    })
    
    return forecast_df


def evaluate_model(model, test_df):
    """
    Evaluate the forecasting model on test data.
    
    Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    
    Args:
        model (sklearn.linear_model.LinearRegression): Trained forecasting model.
        test_df (pd.DataFrame): Test data with 'ds' (datetime) and 'y' (actual) columns.
    
    Returns:
        tuple: (mae, rmse) - Mean Absolute Error and Root Mean Squared Error.
    """
    # Convert test datetime to ordinal
    X_test = test_df['ds'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_actual = test_df['y'].values
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    return mae, rmse


def save_model(model, filepath="models/forecast_model.pkl"):
    """
    Save the trained model to disk using pickle.
    
    Args:
        model: Trained model to save.
        filepath (str): Path to save the model. Default is "models/forecast_model.pkl".
    """
    # Create models directory if it doesn't exist
    model_path = Path(filepath)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def plot_forecast(train_df, test_df, forecast_df):
    """
    Plot training data, test data, and future forecast.
    
    Creates a comprehensive visualization showing historical data and predictions.
    
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y' columns.
        test_df (pd.DataFrame): Test data with 'ds' and 'y' columns.
        forecast_df (pd.DataFrame): Forecast data with 'ds' and 'y_pred' columns.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot training data
    ax.plot(train_df['ds'], train_df['y'], 
            label='Training Data', color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot test data
    ax.plot(test_df['ds'], test_df['y'], 
            label='Test Data (Actual)', color='green', linewidth=2, marker='s', markersize=4)
    
    # Plot forecast
    ax.plot(forecast_df['ds'], forecast_df['y_pred'], 
            label='Forecast (Future)', color='red', linewidth=2, marker='^', 
            markersize=4, linestyle='--')
    
    # Formatting
    ax.set_title('Crime Count Forecast - Monthly Trend', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Crime Count', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display plot
    plt.show()


if __name__ == "__main__":
    """
    Test block to verify all functions work correctly.
    """
    # Load cleaned dataset
    df = pd.read_csv("data/cleaned_chicago_crime.csv")
    
    # Prepare monthly aggregated data
    monthly_df = prepare_monthly_data(df)
    print(f"Monthly data prepared. Shape: {monthly_df.shape}")
    print(f"Date range: {monthly_df['ds'].min()} to {monthly_df['ds'].max()}")
    
    # Split into train and test sets
    train_df, test_df = train_test_split_time_series(monthly_df)
    print(f"\nTrain set: {len(train_df)} months")
    print(f"Test set: {len(test_df)} months")
    
    # Train forecasting model
    model = train_forecast_model(train_df)
    print("\nModel trained successfully.")
    
    # Generate future forecast
    forecast_df = forecast_future(model, monthly_df)
    print(f"Forecast generated for {len(forecast_df)} future months.")
    
    # Evaluate model
    mae, rmse = evaluate_model(model, test_df)
    print("\nModel Evaluation:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    
    # Save model
    save_model(model)
    
    # Plot forecast
    plot_forecast(train_df, test_df, forecast_df)
