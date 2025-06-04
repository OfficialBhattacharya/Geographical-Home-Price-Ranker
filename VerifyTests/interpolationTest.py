import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def calculate_growth_rate(values):
    """Calculate the average growth rate from non-null consecutive values"""
    if len(values) < 2:
        return 0
    
    # Filter out NaN values and get consecutive pairs
    clean_values = []
    for i in range(len(values) - 1):
        if pd.notna(values[i]) and pd.notna(values[i + 1]) and values[i] != 0:
            growth_rate = (values[i + 1] - values[i]) / values[i]
            clean_values.append(growth_rate)
    
    return np.mean(clean_values) if clean_values else 0

def interpolate_with_growth_rate(series):
    """Interpolate missing values using growth rates"""
    series = series.copy()
    
    # Calculate overall growth rate for the series
    growth_rate = calculate_growth_rate(series.values)
    
    # Forward fill using growth rate
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]) and pd.notna(series.iloc[i-1]):
            # Use calculated growth rate for interpolation
            series.iloc[i] = series.iloc[i-1] * (1 + growth_rate)
    
    # Backward fill using growth rate for remaining NaN values
    for i in range(len(series) - 2, -1, -1):
        if pd.isna(series.iloc[i]) and pd.notna(series.iloc[i+1]):
            # Use inverse growth rate for backward interpolation
            series.iloc[i] = series.iloc[i+1] / (1 + growth_rate) if (1 + growth_rate) != 0 else series.iloc[i+1]
    
    # If still NaN values remain, use linear interpolation as fallback
    if series.isna().any():
        series = series.interpolate(method='linear')
    
    # Fill any remaining NaN with forward fill and backward fill
    series = series.fillna(method='ffill').fillna(method='bfill')
    
    return series

def create_interpolated_dataset():
    """Main function to create interpolated dataset from 1990-01-01 to 2025-01-01"""
    
    print("Loading dataset...")
    # Load the original dataset
    df = pd.read_csv('Regression_HPI/USA_MonthlyMacroData.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter data from 1990-01-01 onwards
    start_date = datetime(1990, 1, 1)
    end_date = datetime(2025, 1, 1)
    
    df_filtered = df[df['Date'] >= start_date].copy()
    
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Filtered date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    
    # Create complete monthly date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
    
    print(f"Target date range: {len(date_range)} months from {date_range[0]} to {date_range[-1]}")
    
    # Create new dataframe with complete date range
    new_df = pd.DataFrame({'Date': date_range})
    new_df['Region'] = 'United States'  # Keep the Region column consistent
    
    # Merge with existing data
    df_complete = new_df.merge(df_filtered, on=['Date', 'Region'], how='left')
    
    print(f"Complete dataset shape before interpolation: {df_complete.shape}")
    
    # Get numeric columns (exclude Date and Region)
    numeric_columns = df_complete.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numeric_columns)} numeric columns to interpolate")
    
    # Count missing values before interpolation
    missing_before = df_complete[numeric_columns].isna().sum().sum()
    print(f"Total missing values before interpolation: {missing_before}")
    
    print("Starting interpolation process...")
    
    # Interpolate each numeric column using growth rates
    for col in numeric_columns:
        if df_complete[col].notna().sum() > 1:  # Only interpolate if we have at least 2 non-null values
            print(f"Interpolating column: {col}")
            df_complete[col] = interpolate_with_growth_rate(df_complete[col])
        else:
            print(f"Skipping column {col}: insufficient data")
    
    # Count missing values after interpolation
    missing_after = df_complete[numeric_columns].isna().sum().sum()
    print(f"Total missing values after interpolation: {missing_after}")
    
    # Sort by date to ensure proper ordering
    df_complete = df_complete.sort_values('Date').reset_index(drop=True)
    
    # Save the interpolated dataset
    output_filename = 'USA_MonthlyMacroData_Interpolated_1990_2025.csv'
    df_complete.to_csv(output_filename, index=False)
    
    print(f"\nInterpolated dataset saved as: {output_filename}")
    print(f"Final dataset shape: {df_complete.shape}")
    print(f"Date range: {df_complete['Date'].min()} to {df_complete['Date'].max()}")
    
    # Display summary statistics
    print("\nSummary of interpolation:")
    print(f"- Original missing values: {missing_before}")
    print(f"- Missing values after interpolation: {missing_after}")
    print(f"- Values interpolated: {missing_before - missing_after}")
    
    # Show first and last few rows
    print("\nFirst 5 rows:")
    print(df_complete.head())
    print("\nLast 5 rows:")
    print(df_complete.tail())
    
    # Check for any remaining missing values by column
    missing_by_column = df_complete[numeric_columns].isna().sum()
    if missing_by_column.sum() > 0:
        print("\nColumns with remaining missing values:")
        print(missing_by_column[missing_by_column > 0])
    else:
        print("\nNo missing values remaining in numeric columns!")
    
    return df_complete

if __name__ == "__main__":
    print("Starting macro data interpolation process...")
    print("=" * 60)
    
    try:
        result_df = create_interpolated_dataset()
        print("=" * 60)
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 