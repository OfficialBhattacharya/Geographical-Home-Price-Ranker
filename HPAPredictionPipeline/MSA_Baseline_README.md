# MSA Baseline Model - Updated Script

## Overview

The `msaBaseline_oneShot.py` script has been completely rewritten to handle USA and MSA raw data inputs and produce the specific output format required for the Geographical Home Price Ranker project.

## Key Changes Made

### 1. **Input Data Structure**
- **Before**: Single unified dataset with generic region processing
- **After**: Two separate input files:
  - MSA raw data (contains MSA-specific information)
  - USA raw data (contains USA-level baseline projections)

### 2. **Target Variable**
- **Before**: Generic target column (user-specified)
- **After**: Specifically targets `hpa12m` to predict 1-year forward HPA (`HPA1Yfwd`)

### 3. **Output Format**
- **Before**: Generic region-wise results
- **After**: Standardized output with exact columns required:
  - `Year_Month_Day`: Date in YYYY-MM-01 format
  - `rcode`: MSA region code
  - `cs_name`: MSA region name  
  - `tag`: Train/test indicator
  - `ProjectedHPA1YFwd_USABaseline`: USA baseline projections
  - `ProjectedHPA1YFwd_MSABaseline`: MSA baseline projections (model output)
  - `HPI`: Actual HPI values
  - `hpa12m`: Actual 12-month HPA values
  - `HPA1Yfwd`: 1-year forward HPA (target variable)
  - `HPI1Y_fwd`: 1-year forward HPI
  - `USA_HPA1Yfwd`: USA 1-year forward HPA
  - `USA_HPI1Yfwd`: USA 1-year forward HPI

## Input Data Requirements

### MSA Raw Data Format (`msa_data.csv`):
```csv
Year_Month_Day,rcode,cs_name,HPI,hpa12m,[additional_features...]
2020-01-01,12345,Metro Area 1,250.5,0.045,3.2,75000
2020-02-01,12345,Metro Area 1,252.1,0.047,3.1,75200
2020-01-01,23456,Metro Area 2,180.2,0.038,4.5,65000
...
```

**Required Columns:**
- `Year_Month_Day`: Date column (YYYY-MM-DD format)
- `rcode`: MSA region code (unique identifier)
- `cs_name`: MSA region name
- `HPI`: Housing Price Index values
- `hpa12m`: 12-month Home Price Appreciation

**Optional Columns:**
- Any additional economic indicators (unemployment rate, median income, etc.)

### USA Raw Data Format (`usa_data.csv`):
```csv
Year_Month_Day,ProjectedHPA1YFwd_USABaseline,USA_HPA1Yfwd,USA_HPI1Yfwd
2020-01-01,0.055,0.052,265.2
2020-02-01,0.057,0.054,267.1
2020-03-01,0.056,0.053,268.5
...
```

**Required Columns:**
- `Year_Month_Day`: Date column (YYYY-MM-DD format) 
- `ProjectedHPA1YFwd_USABaseline`: USA baseline projections
- `USA_HPA1Yfwd`: USA 1-year forward HPA
- `USA_HPI1Yfwd`: USA 1-year forward HPI

## Usage

### Method 1: Default Configuration
```python
# Run with default file paths
main()
```

### Method 2: Custom File Paths
```python
# Run with your specific data files
run_with_custom_paths(
    msa_file="path/to/your/msa_data.csv",
    usa_file="path/to/your/usa_data.csv", 
    output_file="path/to/your/output.csv"
)
```

### Method 3: Interactive Configuration
```python
# Get user input for all configurations
cfg = CFG(get_user_input=True)
```

## Model Architecture

### Feature Engineering
1. **Lag Features**: 1, 3, 6, 12, 24 month lags of key variables
2. **Moving Averages**: 3, 6, 12 month moving averages
3. **Rate of Change**: Percentage changes over 1, 3, 6, 12 months
4. **Min/Max Values**: Rolling 12-month minimums and maximums
5. **USA Integration**: Uses USA baseline projections as additional features

### Modeling Approach
1. **Individual Models**: Ridge, Random Forest, XGBoost
2. **Ensemble Method**: Simple averaging of base model predictions
3. **Train/Test Split**: Automatic split where last 12 months are test data (no forward-looking target available)
4. **Cross-Validation**: Time series split for model validation

### Processing Flow
1. Load and merge MSA and USA data
2. Create forward-looking target variables (HPA1Yfwd, HPI1Y_fwd)
3. Generate train/test tags based on target availability
4. Engineer comprehensive feature set
5. Process each MSA region separately
6. Generate ensemble predictions
7. Compile final output with all required columns

## Output Description

The final output CSV contains 12 columns with the following meanings:

- **Year_Month_Day**: Monthly date (always 1st of month)
- **rcode/cs_name**: MSA identifiers
- **tag**: 'train' for historical data with targets, 'test' for projection periods
- **ProjectedHPA1YFwd_USABaseline**: National-level baseline from USA data
- **ProjectedHPA1YFwd_MSABaseline**: MSA-specific model projections (**main output**)
- **HPI/hpa12m**: Actual historical values
- **HPA1Yfwd/HPI1Y_fwd**: 1-year forward actual values (for validation)
- **USA_HPA1Yfwd/USA_HPI1Yfwd**: National forward-looking values

## Performance Characteristics

### Memory and Processing
- **Memory Usage**: Scales with number of MSAs × time periods
- **Processing Time**: ~1-5 minutes per MSA region (depends on data size)
- **Recommended**: Process in batches for datasets with >100 MSAs

### Model Performance
- **Typical R²**: 0.6-0.9 depending on MSA data quality
- **Features Used**: 20-50 engineered features per MSA
- **Validation**: Time series cross-validation prevents data leakage

## Error Handling

The script includes robust error handling for:
- Missing data files
- Insufficient data per MSA (minimum 24 months required)
- Missing required columns
- Date parsing issues
- Model training failures

MSAs with insufficient data or training failures are automatically skipped with detailed logging.

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
```

## Testing

To test the script with sample data:

```python
# Generate sample data and run test
create_sample_data_demo()
run_with_custom_paths('sample_msa_data.csv', 'sample_usa_data.csv', 'sample_output.csv')
```

This creates synthetic data matching the expected format and runs the full pipeline.

## Notes

1. **Data Quality**: Ensure MSA and USA data have consistent date coverage
2. **Missing Values**: The script handles missing values through forward/backward fill and median imputation
3. **Scalability**: For large datasets, consider processing MSAs in batches
4. **Validation**: Compare `ProjectedHPA1YFwd_MSABaseline` with `HPA1Yfwd` for model accuracy assessment 