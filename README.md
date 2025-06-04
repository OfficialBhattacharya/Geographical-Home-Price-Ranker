# FRED Data Processing Pipeline

A comprehensive pipeline for downloading, processing, and cleaning Federal Reserve Economic Data (FRED) series. This system uses a configuration-based approach for easy customization and supports multiple data processing steps from raw data collection to interpolated datasets.

## 🚀 Features

- **Configuration-driven**: All paths, API keys, and settings stored in a single YAML file
- **Batch processing**: Download multiple FRED series automatically
- **Data aggregation**: Combine multiple CSV files into a unified dataset
- **Smart interpolation**: Fill missing values using growth rate analysis
- **Multi-region support**: Handle datasets with multiple geographical regions
- **Error handling**: Robust error handling with retry mechanisms
- **Flexible output**: Configurable file paths and formats

## 📋 Quick Start

### Step 1: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Step 2: Get Your FRED API Key

1. **Visit**: https://fred.stlouisfed.org/docs/api/api_key.html
2. **Click**: "Request an API Key"
3. **Fill out**: Simple registration form (email, name, organization, purpose)
4. **Check email**: API key usually arrives within minutes

### Step 3: Configure the System

1. **Edit `config.yaml`**:
   ```yaml
   api:
     fred_api_key: "YOUR_ACTUAL_API_KEY_HERE"  # Replace with your API key
   
   paths:
     project_root: "YOUR_PROJECT_PATH"  # Update to your project directory
     raw_data_dir: "YOUR_PROJECT_PATH/AllFredFiles"
     processed_data_dir: "YOUR_PROJECT_PATH/AllProcessedFiles"
   ```

2. **Customize FRED Series** (optional):
   - Edit `allFredSeries.txt` to add your own FRED series
   - Format: `SeriesName: https://fred.stlouisfed.org/series/SERIES_ID`

### Step 4: Run the Pipeline

```bash
# Run the complete pipeline (recommended)
python run_pipeline.py

# Or run individual steps
python batch_fred_scraper.py    # Download FRED data
python fredAggregator.py        # Combine into unified dataset
python fredCleaner.py           # Clean and interpolate missing values
```

## 📖 Detailed Setup Guide

### Prerequisites

**Required Python Packages:**
- pandas (data manipulation)
- numpy (numerical operations)
- requests (API calls)
- pyyaml (configuration file parsing)

**System Requirements:**
- Python 3.7 or higher
- Internet connection for FRED API access
- ~50MB disk space for economic data

### FRED API Key Setup

**Why you need it:** The Federal Reserve requires a free API key to access their economic data.

**How to get it:**
1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request an API Key"
3. Fill out the form:
   - **Email**: Your email address
   - **Name**: Your full name
   - **Organization**: Can be "Personal" or "Student"
   - **Purpose**: "Data analysis" or "Research"
4. Check your email for the API key (usually instant)

**Example API key format:** `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

### Configuration Setup

The `config.yaml` file controls all aspects of the pipeline. Here's what each section does:

#### 1. API Configuration
```yaml
api:
  fred_api_key: "YOUR_ACTUAL_API_KEY_HERE"  # Replace with your FRED API key
```

#### 2. Directory Paths
```yaml
paths:
  # Update these paths for your system
  project_root: "C:/your-project-folder"  # Windows
  # project_root: "/home/user/your-project-folder"  # Linux/Mac
  
  raw_data_dir: "C:/your-project-folder/AllFredFiles"
  processed_data_dir: "C:/your-project-folder/AllProcessedFiles"
```

#### 3. Processing Settings
```yaml
processing:
  start_date: "1990-01-01"      # Start date for analysis
  default_end_date: "2025-01-01"  # Default end date (user can override)
```

## 🔄 How to Use - Step by Step

### Method 1: Complete Pipeline (Recommended)

**Single command to do everything:**

```bash
python run_pipeline.py
```

**What happens:**
1. Downloads all FRED series from `allFredSeries.txt`
2. Combines them into a unified monthly dataset
3. Asks for end date (press Enter for default: 2025-01-01)
4. Interpolates missing values using smart growth rate analysis
5. Saves clean dataset ready for analysis

**Expected output:**
```
FRED Data Processing Pipeline
============================================================
[OK] Configuration validated successfully

STEP 1/3: Downloading FRED data
...
[SUCCESS] Successfully downloaded 20 series

STEP 2/3: Aggregating data  
...
[SUCCESS] Created unified dataset with 930 rows, 22 columns

STEP 3/3: Cleaning and interpolating
Enter the end date (YYYY-MM-DD format, default: 2025-01-01): [Press Enter]
...
[SUCCESS] Interpolated 1,620 missing values

PIPELINE COMPLETED SUCCESSFULLY!
```

### Method 2: Step-by-Step Execution

**If you want more control over each step:**

#### Step 1: Download FRED Data
```bash
python batch_fred_scraper.py
```
- Downloads all series listed in `allFredSeries.txt`
- Saves individual CSV files to `AllFredFiles/` directory
- Shows progress for each series

#### Step 2: Aggregate Data
```bash
python fredAggregator.py
```
- Processes all CSV files in `AllFredFiles/`
- Detects data frequency (monthly/quarterly/annual)
- Converts everything to monthly frequency
- Creates `unified_monthly_data.csv`

#### Step 3: Clean and Interpolate
```bash
python fredCleaner.py
```
- Loads the unified dataset
- Prompts for end date
- Fills missing values using growth rate interpolation
- Creates final clean dataset

### Method 3: Single Series Download

**To download just one FRED series:**

```bash
python fredScraper.py "SeriesName: https://fred.stlouisfed.org/series/UNRATE"
```

## 📊 Adding Your Own FRED Series

### Finding FRED Series

1. **Go to**: https://fred.stlouisfed.org/
2. **Search** for economic indicators (e.g., "unemployment rate", "GDP", "inflation")
3. **Click** on the series you want
4. **Copy** the URL (e.g., `https://fred.stlouisfed.org/series/UNRATE`)

### Adding to the Pipeline

**Edit `allFredSeries.txt`:**

```text
# Economic Indicators - Add your series here
# Format: SeriesName: https://fred.stlouisfed.org/series/SERIES_ID

UnemploymentRate: https://fred.stlouisfed.org/series/UNRATE
GrossDomesticProduct: https://fred.stlouisfed.org/series/GDP
InflationRate: https://fred.stlouisfed.org/series/CPIAUCSL
ConsumerPriceIndex: https://fred.stlouisfed.org/series/CPIAUCSL

# Housing Market Indicators
HomePriceIndex: https://fred.stlouisfed.org/series/CSUSHPINSA
HousingSales: https://fred.stlouisfed.org/series/HSN1F

# Add your custom series here:
YourSeriesName: https://fred.stlouisfed.org/series/YOUR_SERIES_ID
```

**Popular FRED Series Examples:**
- **UNRATE**: Unemployment Rate
- **GDP**: Gross Domestic Product  
- **CPIAUCSL**: Consumer Price Index
- **FEDFUNDS**: Federal Funds Rate
- **CSUSHPINSA**: Case-Shiller Home Price Index
- **PAYEMS**: Total Nonfarm Payrolls

## 📁 Understanding Output Files

### Raw Data Files (`AllFredFiles/`)
- **Format**: Individual CSV files per FRED series
- **Naming**: `{SERIES_ID}.csv` (e.g., `UNRATE.csv`)
- **Content**: Date and value columns as downloaded from FRED

### Unified Dataset (`unified_monthly_data.csv`)
- **Format**: All series combined in one file
- **Columns**: Date, Region, plus all economic indicators
- **Frequency**: All data normalized to monthly (first day of month)
- **Use**: Good for initial data exploration

### Clean Dataset (`unified_monthly_data_interpolated_YYYYMMDD.csv`)
- **Format**: Complete dataset with no missing values
- **Date Range**: From start_date to your specified end date
- **Content**: All missing values filled using smart interpolation
- **Use**: Ready for machine learning, regression analysis, forecasting

## ⚙️ Advanced Configuration

### Customizing Date Ranges

**In `config.yaml`:**
```yaml
processing:
  start_date: "2000-01-01"      # Start from year 2000
  default_end_date: "2030-01-01"  # Extend to 2030
```

### Error Handling Settings

```yaml
error_handling:
  max_retries: 5                # Retry failed downloads 5 times
  api_timeout: 60               # Wait 60 seconds for API response
  continue_on_error: true       # Don't stop if some series fail
```

### File Settings

```yaml
files:
  csv_encoding: "utf-8"         # File encoding
  create_backups: true          # Create backup copies
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. "Configuration file not found"
```
[ERROR] Configuration file 'config.yaml' not found
```
**Solution**: Make sure `config.yaml` exists in your project directory

#### 2. "FRED API key not working"
```
[ERROR] API request failed: 400 - Bad Request
```
**Solutions**:
- Check that your API key is correct in `config.yaml`
- Ensure your API key is active (check email from FRED)
- Try requesting a new API key if needed

#### 3. "Path not found" errors
```
[ERROR] Input file not found at [path]
```
**Solution**: Update all paths in `config.yaml` to match your system

#### 4. "No data downloaded"
```
Warning: No valid data points found
```
**Solutions**:
- Verify the FRED series ID is correct
- Check if the series has been discontinued
- Try a different series to test

#### 5. Unicode/Encoding errors (Windows)
**Solution**: The scripts use ASCII characters to avoid encoding issues. If you still see problems, try running in PowerShell instead of Command Prompt.

### Testing Your Setup

**Test configuration:**
```bash
python config_loader.py
# Should show: [OK] Configuration validation passed
```

**Test requirements:**
```bash
python run_pipeline.py --check
# Should show: [OK] All requirements satisfied
```

**Test with single series:**
```bash
python fredScraper.py "TestSeries: https://fred.stlouisfed.org/series/UNRATE"
```

## 🔧 Command Line Options

### Pipeline Runner Options

```bash
python run_pipeline.py              # Run complete pipeline
python run_pipeline.py --help       # Show help
python run_pipeline.py --check      # Check requirements only
python run_pipeline.py --config     # Test configuration only
```

### Individual Script Options

```bash
# Download single series
python fredScraper.py "SeriesName: URL"

# Batch download with custom file
python batch_fred_scraper.py

# Process custom directory
python fredAggregator.py

# Interactive cleaning
python fredCleaner.py
```

## 📈 Example Usage Scenarios

### Scenario 1: Housing Market Analysis
```bash
# 1. Add housing-related series to allFredSeries.txt:
#    HomePrices: https://fred.stlouisfed.org/series/CSUSHPINSA
#    HousingSales: https://fred.stlouisfed.org/series/HSN1F
#    MortgageRates: https://fred.stlouisfed.org/series/MORTGAGE30US

# 2. Run pipeline
python run_pipeline.py
# Enter end date: 2024-12-01

# 3. Output: Clean dataset with monthly housing indicators
```

### Scenario 2: Economic Forecasting
```bash
# 1. Use default economic indicators (unemployment, inflation, etc.)
# 2. Run with extended date range
python run_pipeline.py
# Enter end date: 2026-01-01  # Extend for forecasting

# 3. Use interpolated dataset for time series forecasting
```

### Scenario 3: Custom Research Project
```bash
# 1. Edit allFredSeries.txt with your specific indicators
# 2. Customize config.yaml for your date range and paths
# 3. Run pipeline and get research-ready dataset
```

## 📝 Tips for Success

1. **Start small**: Test with 2-3 series first before adding many
2. **Check data quality**: Review the unified dataset before interpolation
3. **Understand your data**: Different series have different start dates and frequencies
4. **Backup your config**: Save a copy of your working `config.yaml`
5. **Monitor API usage**: FRED has rate limits, but the pipeline handles retries

## 🤝 Contributing

To add new features or modify the pipeline:

1. All configurations should be added to `config.yaml`
2. Use the `config_loader.py` module to access settings
3. Follow the existing error handling patterns
4. Update this README with any new features

## 📄 License

This project is open source. Feel free to modify and distribute according to your needs.

---

**Need help?** Check the troubleshooting section above or review the console output for detailed error messages. 