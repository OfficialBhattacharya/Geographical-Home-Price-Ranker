# State-Level Data Enhancement for FRED Pipeline

This module extends the existing FRED data ingestion framework to include comprehensive state-level economic indicators, providing granular regional data for enhanced home price prediction models.

## 🎯 Overview

The State-Level Data Enhancement module fetches and processes state-specific economic indicators from the FRED API, following the same patterns as national-level data but with state-specific series IDs. This enables:

- **Regional Analysis**: State-by-state economic performance tracking
- **Enhanced Predictions**: More granular input for MSA-level models
- **Geographic Insights**: Regional economic divergence analysis
- **Comprehensive Coverage**: 51 states/territories with multiple economic indicators

## 📊 State-Level Series Coverage

### Available State-Level Indicators

| Series Name | FRED Pattern | Frequency | Coverage | Notes |
|-------------|--------------|-----------|----------|-------|
| AverageSalesPrice_NewHousesSold | ASPNHS[ST] | Monthly | All states | New home sales prices |
| NewOneFamilyHousesSold | HSN1F[ST] | Monthly | All states | New home sales volume |
| MedianSalesPriceofHousesSold | MEDSP[ST] | Monthly | All states | Median home prices |
| HomeownershipRate | EQV[ST]RATEM | Monthly | All states | Homeownership percentage |
| UnemploymentRate | ST[ST]URN | Monthly | All states | State unemployment rates |
| HomePriceIndex | [ST]STHPI | Monthly | All states | FHFA Home Price Index |
| OccupiedHousingUnits | EOCC[ST]Q176N | Quarterly | All states | Housing occupancy |
| VacantHousingUnits_1 | EQVAC[ST] | Quarterly | All states | Total vacant units |
| VacantforOtherReasons | EQVOM[ST] | Quarterly | All states | Other vacant units |
| RenterOccupiedHousingUnits | EQVRENT[ST] | Quarterly | All states | Rental occupancy |
| VacantHousingUnits_NotYetOccupied | EQVSLD[ST] | Quarterly | All states | New vacant units |
| VacantHousingUnits_forSale | EQVFS[ST] | Quarterly | All states | For-sale vacant units |
| TotalHousingUnits | ETOTAL[ST]Q176N | Quarterly | All states | Total housing stock |
| MedianDaysonMarket | MEDDAYONMAR[ST] | Monthly | Limited | Realtor.com data |
| MedianListingPriceperSquareFeet | MEDLISPRIPERSQUFEE[ST] | Monthly | Limited | Realtor.com data |

### Proxy Series

| Series Name | FRED Pattern | Proxy For | Notes |
|-------------|--------------|-----------|-------|
| TotalShipmentsofNewHomes | [ST]BPPRIVSA | TotalShipmentsofNewHomes | Building permits as proxy |

### Unavailable at State Level

- AverageSalesPrice (ASPUS)
- MonthlySupplyofNewHouses (MSACSR)
- EconomicPolicyUncertaintyIndex (USEPUINDXD)

## 🗺️ Geographic Coverage

**51 States/Territories**: AL, AK, AZ, AR, CA, CO, CT, DE, FL, GA, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT, NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VT, VA, WA, WV, WI, WY, DC

## 🔧 Technical Implementation

### Core Components

1. **StateDataEnhancer Class**: Main processing engine
2. **State Mappings File**: Series ID patterns for each state
3. **Batch State Scraper**: Automated processing pipeline
4. **Integration Module**: Combines state and national data

### Data Processing Pipeline

```python
# Initialize enhancer
enhancer = StateDataEnhancer()

# Process all states and series
state_data = enhancer.enhance_dataset('state_level_data.csv')

# Validate data quality
validation_results = enhancer.validate_2023_data(state_data)

# Integrate with national data
integrated_data = enhancer.integrate_with_main_dataset(
    'unified_monthly_data.csv', 
    state_data
)
```

### Frequency Handling

- **Monthly Data**: Preserved as-is
- **Quarterly Data**: Linear interpolation to monthly (except vacancy metrics)
- **Annual Data**: Linear interpolation to monthly
- **Vacancy Metrics**: Maintained as quarterly (no interpolation)

### Special Handling

#### Realtor.com Series
- **Coverage Check**: Minimum 60% data availability in last 2 years
- **Limited States**: Not all states have Realtor.com data
- **Quality Control**: Automatic filtering of low-coverage states

#### Proxy Series
- **Building Permits**: Used as proxy for new home shipments
- **Seasonal Adjustment**: Applied to raw data when needed
- **Tagging**: Marked with `ProxyFor` field for transparency

## 📈 Output Format

### State-Level Data Structure

```csv
Date,State,SeriesID,Value,Frequency,Source,ProxyFor
2023-01-01,CA,CASTHPI,350.2,monthly,FRED,
2023-01-01,CA,STCAURN,4.1,monthly,FRED,
2023-01-01,CA,CABPPRIVSA,1250,monthly,FRED,TotalShipmentsofNewHomes
...
```

### Integrated Dataset Structure

```csv
Date,Region,State,SeriesID,Value,Frequency,Source,ProxyFor
2023-01-01,United States,CA,CASTHPI,350.2,monthly,FRED,
2023-01-01,United States,TX,TXSTHPI,280.5,monthly,FRED,
2023-01-01,United States,NY,NYSTHPI,420.1,monthly,FRED,
...
```

## 🚀 Usage Guide

### Quick Start

1. **Setup Configuration**
   ```bash
   # Ensure FRED API key is set in config.yaml
   api:
     fred_api_key: "YOUR_API_KEY_HERE"
   ```

2. **Run State-Level Enhancement**
   ```bash
   # Run complete state-level processing
   python batch_state_scraper.py
   
   # Or run as part of full pipeline
   python run_pipeline.py
   ```

3. **Check Results**
   ```bash
   # View state-level data
   head -10 state_level_data.csv
   
   # View metadata
   cat state_data_metadata.json
   ```

### Advanced Usage

#### Custom State Processing
```python
from stateDataEnhancer import StateDataEnhancer

# Initialize with custom config
enhancer = StateDataEnhancer()

# Process specific states only
custom_states = ['CA', 'TX', 'NY', 'FL']
enhancer.states = custom_states

# Run enhancement
state_data = enhancer.enhance_dataset('custom_states.csv')
```

#### Integration with Existing Pipeline
```python
# Load existing national data
main_df = pd.read_csv('AllProcessedFiles/unified_monthly_data.csv')

# Add state-level data
integrated_df = enhancer.integrate_with_main_dataset(
    'AllProcessedFiles/unified_monthly_data.csv',
    state_data
)

# Save integrated dataset
integrated_df.to_csv('enhanced_dataset.csv', index=False)
```

## 📊 Data Quality & Validation

### Coverage Statistics

The module tracks comprehensive coverage statistics:

- **Per-State Coverage**: Percentage of successful series per state
- **Per-Series Coverage**: Percentage of successful states per series
- **Overall Coverage**: Average coverage across all states and series

### Validation Checks

1. **2023 Data Validation**: Cross-check against FRED web interface
2. **Value Range Checks**: Ensure unemployment rates (0-20%), HPI (positive)
3. **Frequency Verification**: Confirm expected data frequencies
4. **Coverage Thresholds**: Enforce minimum coverage requirements

### Quality Metrics

```json
{
  "coverage_percentage": {
    "CA": 95.2,
    "TX": 92.8,
    "NY": 89.5
  },
  "successful_series": {
    "CA": ["UnemploymentRate", "HomePriceIndex", ...],
    "TX": ["UnemploymentRate", "HomePriceIndex", ...]
  },
  "failed_series": {
    "CA": ["MedianDaysonMarket"],
    "TX": []
  }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Solution: Automatic retry with exponential backoff
   - Monitor: Check `state_data_metadata.json` for failed requests

2. **Missing State Data**
   - Cause: Some states lack certain series
   - Solution: Automatic filtering and coverage reporting

3. **Realtor.com Coverage**
   - Cause: Limited geographic coverage
   - Solution: Automatic 60% coverage threshold

4. **Frequency Mismatches**
   - Cause: Mixed quarterly/monthly data
   - Solution: Automatic interpolation with metadata tracking

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
enhancer = StateDataEnhancer()
state_data = enhancer.enhance_dataset()
```

## 📈 Performance Considerations

### Processing Time

- **51 States × 15 Series**: ~765 API calls
- **Estimated Time**: 15-30 minutes (with rate limiting)
- **Memory Usage**: ~500MB for full dataset

### Optimization Tips

1. **Parallel Processing**: Consider running states in parallel (with rate limit awareness)
2. **Caching**: Implement local caching for repeated runs
3. **Incremental Updates**: Process only new data since last run

### Rate Limiting

- **FRED API**: 120 requests per minute
- **Automatic Handling**: Built-in retry logic with exponential backoff
- **Monitoring**: Progress tracking and error reporting

## 🔗 Integration with ML Pipeline

### Feature Engineering

State-level data can be used to create:

1. **Regional Features**: State-specific economic indicators
2. **Deviation Metrics**: State vs. national performance
3. **Geographic Clusters**: Regional economic patterns
4. **Temporal Features**: State-specific trends and cycles

### Model Enhancement

```python
# Example: Add state-level features to MSA model
def add_state_features(msa_data, state_data):
    # Get state for each MSA
    msa_to_state = get_msa_state_mapping()
    
    # Add state-level features
    for msa in msa_data:
        state = msa_to_state[msa]
        state_features = state_data[state_data['State'] == state]
        
        # Add unemployment rate
        msa_data[f'{msa}_unemployment'] = state_features[
            state_features['SeriesID'].str.contains('URN')
        ]['Value'].iloc[-1]
        
        # Add home price index
        msa_data[f'{msa}_hpi'] = state_features[
            state_features['SeriesID'].str.contains('STHPI')
        ]['Value'].iloc[-1]
    
    return msa_data
```

## 📚 API Reference

### StateDataEnhancer Class

#### Methods

- `enhance_dataset(output_file=None)`: Main processing method
- `validate_2023_data(df)`: Validate data against FRED web interface
- `integrate_with_main_dataset(main_path, state_data)`: Combine datasets
- `_fetch_fred_series(series_id)`: Fetch single FRED series
- `_process_state_series(state, series_name, pattern)`: Process single state-series

#### Properties

- `state_mappings`: Dictionary of series name to FRED pattern
- `states`: List of state abbreviations
- `metadata`: Processing metadata and statistics

### Configuration

All settings are inherited from the main `config.yaml`:

```yaml
api:
  fred_api_key: "YOUR_API_KEY"
  max_retries: 3
  api_timeout: 30

error_handling:
  continue_on_error: true
```

## 🤝 Contributing

To extend the state-level data enhancement:

1. **Add New Series**: Update `state_level_fred_mappings.txt`
2. **Add New States**: Update state abbreviations list
3. **Custom Processing**: Extend `StateDataEnhancer` class
4. **Validation**: Add new validation rules

## 📄 License

This module follows the same license as the main project. 