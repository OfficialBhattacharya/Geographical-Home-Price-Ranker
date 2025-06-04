import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
from config_loader import get_config

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

def get_end_date_from_user(config):
    """Get end date from user input with default from config"""
    default_end_date = config.get_default_end_date()
    
    while True:
        try:
            prompt = f"Enter the end date (YYYY-MM-DD format, default: {default_end_date}): "
            end_date_str = input(prompt).strip()
            
            # Use default if no input provided
            if not end_date_str:
                end_date_str = default_end_date
                
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            print(f"End date set to: {end_date.strftime('%Y-%m-%d')}")
            return end_date
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-01-01)")

def create_interpolated_dataset():
    """Main function to create interpolated dataset from start date to user-specified end date"""
    
    try:
        # Load and validate configuration
        config = get_config()
        config.validate_config()
        
        # Get paths from config
        input_file = config.get_unified_data_path()
        output_dir = config.get_processed_data_dir()
        
        print("Loading dataset...")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found at {input_file}")
            print("Please run the aggregator first to create the unified dataset.")
            return None
        
        # Load the original dataset
        encoding = config.get_file_setting('csv_encoding')
        df = pd.read_csv(input_file, encoding=encoding)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get end date from user
        end_date = get_end_date_from_user(config)
        
        # Get start date from config
        start_date_str = config.get_start_date()
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        df_filtered = df[df['Date'] >= start_date].copy()
        
        print(f"Filtered dataset shape: {df_filtered.shape}")
        print(f"Filtered date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
        
        # Create complete monthly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
        
        print(f"Target date range: {len(date_range)} months from {date_range[0]} to {date_range[-1]}")
        
        # Create new dataframe with complete date range
        new_df = pd.DataFrame({'Date': date_range})
        
        # Check if Region column exists in the original data
        if 'Region' in df_filtered.columns:
            # Get unique regions from the original data
            unique_regions = df_filtered['Region'].unique()
            print(f"Found {len(unique_regions)} unique regions: {unique_regions}")
            
            # Create complete dataset for all regions
            complete_data = []
            for region in unique_regions:
                region_df = new_df.copy()
                region_df['Region'] = region
                complete_data.append(region_df)
            
            new_df_complete = pd.concat(complete_data, ignore_index=True)
        else:
            # If no Region column, assume single region
            new_df_complete = new_df.copy()
            new_df_complete['Region'] = 'United States'  # Default region name
        
        # Merge with existing data
        if 'Region' in df_filtered.columns:
            df_complete = new_df_complete.merge(df_filtered, on=['Date', 'Region'], how='left')
        else:
            df_complete = new_df_complete.merge(df_filtered, on=['Date'], how='left')
        
        print(f"Complete dataset shape before interpolation: {df_complete.shape}")
        
        # Get numeric columns (exclude Date and Region)
        numeric_columns = df_complete.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Found {len(numeric_columns)} numeric columns to interpolate")
        
        # Count missing values before interpolation
        missing_before = df_complete[numeric_columns].isna().sum().sum()
        print(f"Total missing values before interpolation: {missing_before}")
        
        print("Starting interpolation process...")
        
        # If we have multiple regions, interpolate each region separately
        if 'Region' in df_complete.columns and len(df_complete['Region'].unique()) > 1:
            interpolated_dfs = []
            for region in df_complete['Region'].unique():
                print(f"Processing region: {region}")
                region_df = df_complete[df_complete['Region'] == region].copy()
                
                # Interpolate each numeric column using growth rates
                for col in numeric_columns:
                    if region_df[col].notna().sum() > 1:  # Only interpolate if we have at least 2 non-null values
                        print(f"  Interpolating column: {col}")
                        region_df[col] = interpolate_with_growth_rate(region_df[col])
                    else:
                        print(f"  Skipping column {col}: insufficient data")
                
                interpolated_dfs.append(region_df)
            
            # Combine all regions
            df_complete = pd.concat(interpolated_dfs, ignore_index=True)
        else:
            # Single region processing
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
        if 'Region' in df_complete.columns:
            df_complete = df_complete.sort_values(['Region', 'Date']).reset_index(drop=True)
        else:
            df_complete = df_complete.sort_values('Date').reset_index(drop=True)
        
        # Create output filename with end date
        end_date_str = end_date.strftime('%Y-%m-%d')
        base_filename = config.get_path('unified_data_file').replace('.csv', '_interpolated.csv')
        output_filename = config.create_output_filename(base_filename, end_date_str)
        output_path = os.path.join(output_dir, os.path.basename(output_filename))
        
        # Save the interpolated dataset
        df_complete.to_csv(output_path, index=False, encoding=encoding)
        
        print(f"\n[SUCCESS] Interpolated dataset saved as: {output_path}")
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
            print("\n[SUCCESS] No missing values remaining in numeric columns!")
        
        return df_complete
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting macro data interpolation process...")
    print("=" * 60)
    
    try:
        result_df = create_interpolated_dataset()
        if result_df is not None:
            print("=" * 60)
            print("[SUCCESS] Process completed successfully!")
        else:
            print("=" * 60)
            print("[ERROR] Process failed!")
            
    except Exception as e:
        print(f"[ERROR] Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 