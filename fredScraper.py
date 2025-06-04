import os
import requests
import csv
import sys
from config_loader import get_config

def download_fred_series(input_str, output_dir=None):
    """Download a FRED series and save to CSV file"""
    
    # Load configuration
    config = get_config()
    
    # Use output_dir from config if not provided
    if output_dir is None:
        output_dir = config.get_raw_data_dir()
    
    # Parse input string
    if ': ' not in input_str:
        raise ValueError("Input format must be 'SeriesName: URL'")
    series_name, url = input_str.split(': ', 1)
    
    # Extract series ID from URL
    series_id = url.rstrip('/').split('/')[-1].rstrip(' ,')
    
    # Get API key from config
    api_key = config.get_api_key()
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not found in configuration file")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # API endpoint with timeout and retry settings
    api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    
    # Get settings from config
    max_retries = config.get_max_retries()
    timeout = config.get_api_timeout()
    
    # Fetch data with retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, timeout=timeout)
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limit
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise Exception(f"API request timed out after {max_retries} attempts")
            continue
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"API request failed: {str(e)}")
            continue
    
    data = response.json()
    observations = data.get('observations', [])
    if not observations:
        raise ValueError("No observations found in the response")
    
    # Filter out missing values and prepare CSV data
    csv_data = []
    for obs in observations:
        if obs['value'] != '.':  # FRED uses '.' for missing values
            csv_data.append({'Date': obs['date'], series_name: obs['value']})
    
    if not csv_data:
        raise ValueError("No valid data points found (all values are missing)")
    
    # Write to CSV in the specified directory
    filename = os.path.join(output_dir, f"{series_id}.csv")
    
    # Get file encoding from config
    encoding = config.get_file_setting('csv_encoding')
    
    with open(filename, 'w', newline='', encoding=encoding) as csvfile:
        fieldnames = ['Date', series_name]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    return filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fredScraper.py \"SeriesName: URL\"")
        print("Example: python fredScraper.py \"AverageSalesPrice: https://fred.stlouisfed.org/series/ASPUS\"")
        sys.exit(1)
    
    input_str = sys.argv[1]
    try:
        # Validate configuration first
        config = get_config()
        config.validate_config()
        
        output_file = download_fred_series(input_str)
        print(f"[SUCCESS] Data saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)