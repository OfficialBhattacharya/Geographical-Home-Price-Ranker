# MSA-Level Data Enhancement for FRED Pipeline

This module extends the existing FRED data ingestion framework to include comprehensive MSA-level (Metropolitan Statistical Area) economic indicators, providing the finest granular regional data for enhanced home price prediction models across the top 150 metropolitan areas in the United States.

## 🎯 Overview

The MSA-Level Data Enhancement module fetches and processes metropolitan area-specific economic indicators from the FRED API, completing the geographical hierarchy from National → State → MSA levels. This enables:

- **Metropolitan Analysis**: MSA-by-MSA economic performance tracking for major urban centers
- **Hyperlocal Predictions**: Most granular input for neighborhood-level models
- **Urban Economic Insights**: Metropolitan area economic divergence and convergence analysis  
- **Comprehensive Geographic Coverage**: 150+ MSAs covering 80%+ of US economic activity

## 📊 MSA-Level Series Coverage

### Available MSA-Level Indicators

| Series Name | FRED Pattern | Frequency | Coverage | Notes |
|-------------|--------------|-----------|----------|-------|
| UnemploymentRate | [CBSA]UR | Monthly | Major MSAs | Metropolitan unemployment rates |
| HomePriceIndex | ATNHPIUS[CBSA]Q | Quarterly | Major MSAs | FHFA All-Transactions HPI |
| MedianDaysonMarket | MEDDAYONMAR[CBSA] | Monthly | Limited | Realtor.com market timing data |
| MedianListingPriceperSquareFeet | MEDLISPRIPERSQUFEE[CBSA] | Monthly | Limited | Realtor.com pricing data |

### Processing Notes

- **HPI Series**: Quarterly data automatically interpolated to monthly using linear interpolation
- **Realtor.com Data**: Limited coverage - only MSAs with sufficient data density included
- **Coverage Threshold**: Minimum 60% data availability in last 2 years required
- **Geographic Scope**: Top 150 MSAs by population, covering major metropolitan areas

### Unavailable at MSA Level

Most housing and economic series are not available at MSA granularity:

- AverageSalesPrice_NewHousesSold
- NewOneFamilyHousesSold  
- MedianSalesPriceofHousesSold
- HomeownershipRate
- TotalShipmentsofNewHomes
- MonthlySupplyofNewHouses
- EconomicPolicyUncertaintyIndex
- All vacancy and housing unit metrics

## 🗺️ Geographic Coverage

**Top 150 Metropolitan Statistical Areas** including:

### Tier 1 (Population > 4M)
- New York-Newark-Jersey City, NY-NJ-PA (35620)
- Los Angeles-Long Beach-Anaheim, CA (31080) 
- Chicago-Naperville-Elgin, IL-IN-WI (16980)
- Dallas-Fort Worth-Arlington, TX (19100)
- Houston-The Woodlands-Sugar Land, TX (26420)

### Tier 2 (Population 1-4M)  
- Philadelphia, Boston, Phoenix, Seattle, Miami, Denver, Austin, Charlotte, etc.

### Tier 3 (Population 500K-1M)
- Smaller metropolitan areas providing comprehensive regional coverage

*Complete list available in `msa_list_top150.txt`*

## 🔧 Technical Implementation

### Core Components

1. **MSADataEnhancer Class**: Main processing engine for MSA data
2. **MSA Reference Files**: CBSA codes and names mapping
3. **Batch MSA Scraper**: Automated processing pipeline
4. **Data Integration Module**: Combines National/State/MSA data hierarchically

### Data Processing Pipeline

```python
# Initialize MSA enhancer
enhancer = MSADataEnhancer()

# Process all MSAs and available series
msa_data = enhancer.enhance_dataset('msa_level_data.csv')

# Validate data quality
validation_results = enhancer.validate_2023_data(msa_data)

# Integrate with existing datasets
integrated_data = enhancer.integrate_with_main_dataset(
    'state_level_data.csv', 
    msa_data
)
```

### Frequency Handling

- **Monthly Data**: Preserved as-is (UnemploymentRate, Realtor.com series)
- **Quarterly Data**: Linear interpolation to monthly (HomePriceIndex)
- **Coverage Filtering**: Automatic exclusion of MSAs with <60% recent data coverage

### Special Processing

#### Quarterly HPI Interpolation
- FHFA Home Price Index provided quarterly
- Linear interpolation between quarters for monthly estimates
- Marked with `Frequency: 'monthly_interpolated'` for transparency

#### Realtor.com Series Coverage
- **Limited Availability**: Not all MSAs have Realtor.com data
- **Recency Check**: Series older than 6 months automatically excluded
- **Quality Control**: Coverage percentage tracked in metadata

#### Geographic Hierarchy
- **MSA-to-State Mapping**: Automatically extracted from MSA names
- **Hierarchical Integration**: Enables drill-down from National → State → MSA
- **Cross-Reference Validation**: MSA unemployment vs state unemployment consistency checks

## 📈 Output Format

### MSA-Level Data Structure

```csv
Date,CBSA_Code,MSA_Name,SeriesName,Value,Frequency,Source,GeographyType,GeographyCode
2023-01-01,35620,"New York-Newark-Jersey City, NY-NJ-PA",UnemploymentRate,4.1,monthly,FRED,MSA,35620
2023-01-01,35620,"New York-Newark-Jersey City, NY-NJ-PA",HomePriceIndex,285.4,monthly_interpolated,FRED,MSA,35620
2023-01-01,31080,"Los Angeles-Long Beach-Anaheim, CA",UnemploymentRate,5.2,monthly,FRED,MSA,31080
...
```

### Integrated National-State-MSA Dataset Structure

```csv
Date,GeographyType,GeographyCode,GeographyName,State,ParentGeography,SeriesName,Value,Source
2023-01-01,National,US,United States,,,UnemploymentRate,3.5,FRED
2023-01-01,State,CA,California,,US,UnemploymentRate,4.1,FRED
2023-01-01,MSA,31080,"Los Angeles-Long Beach-Anaheim, CA",CA,CA,UnemploymentRate,5.2,FRED
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

2. **Verify Reference Files**
   ```bash
   # Check MSA reference files exist
   ls msa_list_top150.txt msa_level_fred_mappings.txt
   ```

3. **Run MSA-Level Enhancement**
   ```bash
   # Interactive mode with processing options
   python batch_msa_scraper.py
   
   # Or as part of complete pipeline
   python run_pipeline.py
   ```

4. **Validate Results**
   ```bash
   # Run comprehensive validation suite
   python test_msa_enhancement.py
   ```

### Processing Options

The batch MSA scraper provides flexible processing options:

```
Processing Options:
1. Process all 150 MSAs (full dataset) 
2. Process top 50 MSAs only (faster)
3. Process top 10 MSAs only (testing)
4. Custom MSA count

Output Options:
1. MSA-level data only (msa_level_data.csv)
2. Integrated with existing data (integrated_national_state_msa_data.csv)  
3. Both
```

### Advanced Usage

#### Custom MSA Processing
```python
from msaDataEnhancer import MSADataEnhancer

# Initialize with custom config
enhancer = MSADataEnhancer()

# Process specific MSAs only
target_msas = ['35620', '31080', '16980']  # NYC, LA, Chicago
enhancer.msa_reference = enhancer.msa_reference[
    enhancer.msa_reference['CBSA'].isin(target_msas)
]

# Run enhancement
msa_data = enhancer.enhance_dataset('custom_msa_data.csv')
```

#### Data Quality Analysis
```python
# Check coverage by MSA size tier
large_msas = ['35620', '31080', '16980'] 
medium_msas = ['12420', '16740', '19740']
small_msas = ['29620', '22180', '23540']

for tier, msas in [('Large', large_msas), ('Medium', medium_msas), ('Small', small_msas)]:
    tier_data = msa_data[msa_data['CBSA_Code'].isin(msas)]
    coverage = tier_data['SeriesName'].value_counts()
    print(f"{tier} MSAs series coverage: {coverage}")
```

## 📊 Data Quality & Validation

### Validation Framework

The MSA enhancement includes comprehensive validation:

```bash
# Run validation suite
python test_msa_enhancement.py
```

**Validation Components:**
- **File Validation**: Reference file format and completeness
- **API Connectivity**: FRED API access for major MSAs
- **Data Processing**: End-to-end pipeline functionality
- **Spot Checks**: Data availability across MSA size tiers
- **Integration Testing**: Compatibility with existing datasets

### Expected Coverage Rates

| MSA Tier | UnemploymentRate | HomePriceIndex | Realtor.com Data |
|----------|------------------|----------------|------------------|
| Large (>4M) | 95%+ | 90%+ | 70%+ |
| Medium (1-4M) | 85%+ | 75%+ | 45%+ |
| Small (500K-1M) | 70%+ | 50%+ | 25%+ |

### Data Quality Checks

- **Recency Validation**: Latest data within 6 months for active series
- **Coverage Thresholds**: Minimum 60% data availability in last 2 years
- **Value Range Validation**: Unemployment rates 0-15%, HPI positive values
- **Hierarchical Consistency**: MSA unemployment correlated with state rates

## 🔄 Integration with Existing Pipeline

### Pipeline Steps Enhancement

The MSA enhancement integrates seamlessly into the existing pipeline:

```
1. Download National FRED Data (existing)
2. Aggregate National Data (existing)  
3. Clean and Interpolate (existing)
4. State-Level Enhancement (existing)
5. MSA-Level Enhancement (NEW)
6. Hierarchical Integration (NEW)
```

### File Structure

```
/project-root/
├── msa_list_top150.txt              # MSA reference data
├── msa_level_fred_mappings.txt      # Series mapping patterns
├── msaDataEnhancer.py               # Core MSA processing
├── batch_msa_scraper.py             # Batch processing script
├── test_msa_enhancement.py          # Validation suite
├── dataIntegrator.py                # Hierarchical integration
└── /AllProcessedFiles/
    ├── msa_level_data.csv           # MSA-only dataset
    ├── msa_data_metadata.json       # Processing metadata  
    ├── integrated_national_state_msa_data.csv  # Full hierarchy
    └── integration_metadata.json    # Integration summary
```

### Backward Compatibility

- **Existing Functionality**: All current national/state processing unchanged
- **Optional Enhancement**: MSA processing can be skipped if reference files missing
- **Data Formats**: Consistent with existing state-level data structure

## 🎯 Use Cases & Applications

### Home Price Prediction Models

```python
# Load integrated dataset for modeling
df = pd.read_csv('integrated_national_state_msa_data.csv')

# Metropolitan-level home price modeling
msa_hpi = df[
    (df['GeographyType'] == 'MSA') & 
    (df['SeriesName'] == 'HomePriceIndex')
]

# Cross-MSA analysis
unemployment_msa = df[
    (df['GeographyType'] == 'MSA') & 
    (df['SeriesName'] == 'UnemploymentRate')
]
```

### Geographic Hierarchy Analysis

```python
# Compare unemployment across geography levels
national_ur = df[(df['GeographyType'] == 'National') & (df['SeriesName'] == 'UnemploymentRate')]
state_ur = df[(df['GeographyType'] == 'State') & (df['SeriesName'] == 'UnemploymentRate')]  
msa_ur = df[(df['GeographyType'] == 'MSA') & (df['SeriesName'] == 'UnemploymentRate')]

# Hierarchical variance analysis
variance_analysis = {
    'National': national_ur['Value'].std(),
    'State': state_ur.groupby('GeographyCode')['Value'].std().mean(), 
    'MSA': msa_ur.groupby('GeographyCode')['Value'].std().mean()
}
```

### Regional Economic Research

```python
# MSA economic performance clustering
msa_metrics = df[df['GeographyType'] == 'MSA'].pivot_table(
    index=['GeographyCode', 'GeographyName'], 
    columns='SeriesName',
    values='Value',
    aggfunc='mean'
)

# Regional correlation analysis
from scipy.cluster.hierarchy import linkage, dendrogram
msa_clusters = linkage(msa_metrics.fillna(0), method='ward')
```

## 🚨 Limitations & Considerations

### Data Availability Constraints

- **Limited Series**: Only 4 core series available at MSA level vs 15+ at state level
- **Geographic Coverage**: Not all 380+ MSAs have data; focus on top 150 by population
- **Realtor.com Dependency**: Market timing data subject to third-party data policies
- **Quarterly HPI**: Interpolation introduces synthetic monthly values

### Technical Considerations

- **API Rate Limits**: Processing 150 MSAs × 4 series = 600 API calls
- **Processing Time**: Full MSA enhancement takes 10-15 minutes
- **Storage Requirements**: MSA data adds ~50-100MB to existing datasets
- **Memory Usage**: Peak memory usage ~2GB during processing

### Methodological Notes

- **Interpolation Assumptions**: Linear quarterly-to-monthly assumes smooth trends
- **Coverage Bias**: MSAs with better data coverage may not be representative
- **Hierarchy Mapping**: MSA-to-state assignment uses primary state only
- **Seasonal Adjustment**: MSA series may have different seasonal patterns than national/state

## 📞 Support & Troubleshooting

### Common Issues

**"No MSA data retrieved"**
- Check FRED API key validity
- Verify MSA reference files exist and are properly formatted
- Confirm internet connectivity

**"Low coverage warnings"**  
- Normal for smaller MSAs and Realtor.com series
- Adjust `min_coverage_threshold` if needed
- Review metadata for specific coverage details

**"Integration failures"**
- Ensure state-level data processed first
- Check for consistent date formats across datasets
- Verify GeographyType columns properly set

### Performance Optimization

```python
# For faster testing, use smaller MSA subset
enhancer.msa_reference = enhancer.msa_reference.head(10)

# Skip Realtor.com series if coverage issues
enhancer.msa_mappings = {k: v for k, v in enhancer.msa_mappings.items() 
                        if k not in ['MedianDaysonMarket', 'MedianListingPriceperSquareFeet']}
```

### Validation Commands

```bash
# Pre-processing checks
python test_msa_enhancement.py

# Post-processing validation  
python -c "
import pandas as pd
df = pd.read_csv('AllProcessedFiles/msa_level_data.csv')
print(f'MSAs: {df[\"CBSA_Code\"].nunique()}')
print(f'Series: {df[\"SeriesName\"].nunique()}') 
print(f'Records: {len(df):,}')
"
```

## 📚 References & Resources

- **FRED API Documentation**: https://fred.stlouisfed.org/docs/api/
- **CBSA Definitions**: https://www.census.gov/programs-surveys/metro-micro.html
- **FHFA HPI Methodology**: https://www.fhfa.gov/DataTools/Downloads/Pages/House-Price-Index.aspx
- **Realtor.com Data**: https://www.realtor.com/research/data/

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Compatibility**: Python 3.7+, Pandas 1.3+

For questions or issues, please review the validation output in `msa_validation_results.json` and processing logs in `msa_data_metadata.json`. 