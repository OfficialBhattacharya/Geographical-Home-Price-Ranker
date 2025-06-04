import subprocess
import sys
import os
from config_loader import get_config

def check_prerequisites():
    """
    Check if all prerequisites are met before processing
    """
    try:
        # Load and validate configuration
        config = get_config()
        config.validate_config()
        
        # Check if fredScraper.py exists
        if not os.path.exists('fredScraper.py'):
            print("[ERROR] fredScraper.py not found in current directory!")
            return False
        
        # Check if FRED series file exists
        series_file = config.get_fred_series_file()
        if not os.path.exists(series_file):
            print(f"[ERROR] FRED series file not found at {series_file}")
            print("Please ensure allFredSeries.txt exists and contains FRED series information.")
            return False
        
        print("[OK] Prerequisites check passed")
        api_key = config.get_api_key()
        print(f"[OK] API key found: {api_key[:8]}...{api_key[-4:]}")
        print(f"[OK] Raw data directory: {config.get_raw_data_dir()}")
        print(f"[OK] FRED series file: {series_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nTo fix this:")
        print("1. Ensure config.yaml exists in the project directory")
        print("2. Set your FRED API key in config.yaml under 'api.fred_api_key'")
        print("3. Verify all paths in config.yaml are correct")
        return False

def process_all_fred_series():
    """
    Read all lines from the FRED series file and process each one through fredScraper.py
    """
    try:
        config = get_config()
        input_file = config.get_fred_series_file()
        
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        total_series = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        print(f"Found {total_series} series to process")
        
        success_count = 0
        failed_series = []
        continue_on_error = config.should_continue_on_error()
        
        for i, line in enumerate(lines, 1):
            # Clean the line - remove trailing comma, whitespace, and newlines
            cleaned_line = line.strip().rstrip(',').strip()
            
            # Skip empty lines and comments
            if not cleaned_line or cleaned_line.startswith('#'):
                continue
                
            series_name = cleaned_line.split(':')[0]
            print(f"\nProcessing {success_count + len(failed_series) + 1}/{total_series}: {series_name}")
            
            try:
                # Call fredScraper.py with the cleaned line
                result = subprocess.run([
                    sys.executable, "fredScraper.py", cleaned_line
                ], capture_output=True, text=True, check=True, timeout=config.get_api_timeout() + 10)
                
                print(f"[SUCCESS] {result.stdout.strip()}")
                success_count += 1
                
            except subprocess.TimeoutExpired:
                error_msg = f"Request timed out after {config.get_api_timeout() + 10} seconds"
                print(f"[ERROR] {error_msg}")
                failed_series.append(series_name)
                if not continue_on_error:
                    break
                    
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else "Unknown error"
                if e.stdout:
                    error_msg += f" | Output: {e.stdout.strip()}"
                print(f"[ERROR] {error_msg}")
                failed_series.append(series_name)
                if not continue_on_error:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Unexpected error: {str(e)}")
                failed_series.append(series_name)
                if not continue_on_error:
                    break
        
        # Summary
        print(f"\n{'='*50}")
        print(f"SUMMARY:")
        print(f"Total series: {total_series}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {len(failed_series)}")
        
        if failed_series:
            print(f"\nFailed series:")
            for series in failed_series:
                print(f"  - {series}")
        
        return success_count, failed_series
        
    except Exception as e:
        print(f"[ERROR] Error reading series file: {e}")
        return 0, []

if __name__ == "__main__":
    print("Batch FRED Data Scraper")
    print("=" * 50)
    
    # Check prerequisites first
    if not check_prerequisites():
        sys.exit(1)
    
    success_count, failed_series = process_all_fred_series()
    
    if success_count > 0:
        config = get_config()
        print(f"\n[SUCCESS] Successfully created {success_count} CSV files in {config.get_raw_data_dir()}")
    
    if failed_series:
        sys.exit(1)
    else:
        print("[SUCCESS] All series processed successfully!") 