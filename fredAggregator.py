import pandas as pd
import os
from datetime import datetime
import glob
from config_loader import get_config

def determine_frequency(dates, config):
    """Determine if data is monthly, quarterly, or annual based on date patterns"""
    if len(dates) < 2:
        return 'unknown'
    
    # Get thresholds from config
    thresholds = config.get_frequency_thresholds()
    
    # Calculate time differences
    time_diffs = []
    for i in range(1, min(len(dates), 5)):  # Check first few differences
        diff = (dates[i] - dates[i-1]).days
        time_diffs.append(diff)
    
    avg_diff = sum(time_diffs) / len(time_diffs)
    
    if avg_diff <= thresholds['monthly_max_days']:  # Monthly
        return 'monthly'
    elif thresholds['quarterly_min_days'] <= avg_diff <= thresholds['quarterly_max_days']:  # Quarterly
        return 'quarterly'
    elif avg_diff >= thresholds['annual_min_days']:  # Annual
        return 'annual'
    else:
        return 'unknown'

def expand_to_monthly(df, column_name, frequency):
    """Expand quarterly or annual data to monthly - only first day of month"""
    if frequency == 'monthly':
        # For monthly data, ensure we only have first day of month
        monthly_data = []
        for _, row in df.iterrows():
            date = pd.to_datetime(row['Date'])
            # Always use first day of the month
            first_day = date.replace(day=1)
            value = row[column_name]
            monthly_data.append({
                'Date': first_day,
                column_name: value
            })
        return pd.DataFrame(monthly_data)
    
    monthly_data = []
    
    for _, row in df.iterrows():
        date = pd.to_datetime(row['Date'])
        value = row[column_name]
        
        if frequency == 'quarterly':
            # For quarterly data, replicate value for 3 months (first day only)
            if date.month in [1, 4, 7, 10]:  # Quarter start months
                base_month = date.month
                base_year = date.year
                for i in range(3):
                    month_num = base_month + i
                    year = base_year
                    if month_num > 12:
                        month_num -= 12
                        year += 1
                    month_date = datetime(year, month_num, 1)  # Always first day
                    monthly_data.append({
                        'Date': month_date,
                        column_name: value
                    })
                    
        elif frequency == 'annual':
            # For annual data, replicate value for 12 months (first day only)
            year = date.year
            for month in range(1, 13):
                month_date = datetime(year, month, 1)  # Always first day
                monthly_data.append({
                    'Date': month_date,
                    column_name: value
                })
    
    return pd.DataFrame(monthly_data)

def process_csv_files(folder_path, config):
    """Process all CSV files and create unified dataset"""
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_data = {}  # Dictionary to store data from all files
    continue_on_error = config.should_continue_on_error()
    encoding = config.get_file_setting('csv_encoding')
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            # Read CSV file with specified encoding
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Get the data column name (should be the second column)
            data_columns = [col for col in df.columns if col != 'Date']
            if not data_columns:
                print(f"Warning: No data column found in {filename}")
                if not continue_on_error:
                    raise ValueError(f"No data column found in {filename}")
                continue
                
            data_column = data_columns[0]  # Take the first data column
            
            # Determine frequency
            frequency = determine_frequency(df['Date'].tolist(), config)
            print(f"  - Detected frequency: {frequency}")
            
            # Expand to monthly if needed
            monthly_df = expand_to_monthly(df, data_column, frequency)
            
            # Store the monthly data - ensure only first day of month
            for _, row in monthly_df.iterrows():
                date = row['Date']
                # Ensure it's the first day of the month
                first_day = date.replace(day=1)
                date_key = first_day.strftime('%Y-%m-01')  # Always use 01 for day
                
                if date_key not in all_data:
                    all_data[date_key] = {'Date': first_day}
                all_data[date_key][data_column] = row[data_column]
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            if not continue_on_error:
                raise
            continue
    
    # Convert to DataFrame
    if not all_data:
        print("No data was processed successfully")
        return None
        
    result_df = pd.DataFrame.from_dict(all_data, orient='index')
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    
    # Add Region column
    result_df['Region'] = 'United States'
    
    # Reorder columns to have Date and Region first
    columns = ['Date', 'Region'] + [col for col in result_df.columns if col not in ['Date', 'Region']]
    result_df = result_df[columns]
    
    print(f"Final dataset shape: {result_df.shape}")
    print(f"Date format verification - first 5 dates:")
    print(result_df['Date'].head().dt.strftime('%Y-%m-%d').tolist())
    
    return result_df

def main():
    """Main function to run the aggregation process"""
    print("Starting CSV processing...")
    print("=" * 50)
    
    try:
        # Load and validate configuration
        config = get_config()
        config.validate_config()
        
        # Get paths from config
        folder_path = config.get_raw_data_dir()
        output_path = config.get_unified_data_path()
        
        print(f"Source folder: {folder_path}")
        print(f"Output file: {output_path}")
        
        # Check if source folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Source directory not found: {folder_path}")
        
        # Process all CSV files
        unified_df = process_csv_files(folder_path, config)
        
        if unified_df is not None:
            # Save the unified dataset
            encoding = config.get_file_setting('csv_encoding')
            unified_df.to_csv(output_path, index=False, encoding=encoding)
            
            print(f"\n[SUCCESS] Unified dataset created successfully!")
            print(f"Output file: {output_path}")
            print(f"Shape: {unified_df.shape}")
            print(f"Date range: {unified_df['Date'].min()} to {unified_df['Date'].max()}")
            print(f"Columns: {list(unified_df.columns)}")
            
            # Display first few rows
            print("\nFirst 5 rows:")
            print(unified_df.head())
            
            # Display data info
            print(f"\nDataset info:")
            print(f"Total rows: {len(unified_df)}")
            print(f"Total columns: {len(unified_df.columns)}")
            print(f"Missing values per column:")
            missing_values = unified_df.isnull().sum()
            for col, missing in missing_values.items():
                if missing > 0:
                    print(f"  {col}: {missing}")
            
            if missing_values.sum() == 0:
                print("  No missing values found!")
                
        else:
            print("[ERROR] Failed to create unified dataset")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 