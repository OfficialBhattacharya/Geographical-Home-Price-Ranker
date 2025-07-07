# Geographical Home Price Ranker - Complete Pipeline

A comprehensive three-step machine learning pipeline for projecting 1-year forward Home Price Appreciation (HPA) across Metropolitan Statistical Areas (MSAs) using USA baseline projections and regional economic indicators.

## 🎯 Overview

This pipeline implements a hierarchical forecasting approach that combines national-level economic trends with MSA-specific characteristics to generate accurate 1-year forward HPA projections. The methodology follows a three-step process:

1. **USA Baseline Projection**: National-level HPA forecasting using macroeconomic indicators
2. **MSA Baseline Projection**: Regional HPA modeling incorporating USA baseline as a feature
3. **MSA Calibration**: Fine-tuning regional projections using latest economic data

## 📊 Mathematical Foundation

### Home Price Appreciation (HPA) Definition

HPA is the percentage change in housing prices over a specified period:

$$\text{HPA}_{t,t+k} = \frac{\text{HPI}_{t+k} - \text{HPI}_t}{\text{HPI}_t} \times 100$$

Where:
- $\text{HPI}_t$ = Housing Price Index at time $t$
- $\text{HPI}_{t+k}$ = Housing Price Index at time $t+k$
- $k$ = forecast horizon (12 months for 1-year forward)

### 1-Year Forward HPA Target Variable

The target variable for our models is:

$$\text{HPA1Yfwd}_t = \frac{\text{HPI}_{t+12} - \text{HPI}_t}{\text{HPI}_t} \times 100$$

This represents the 12-month forward HPA that we aim to predict at each time point $t$.

## 🔄 Three-Step Pipeline Architecture

### Step 1: USA Baseline Projection

**Objective**: Generate national-level HPA projections using macroeconomic indicators

**Mathematical Model**:
$$\text{ProjectedHPA1YFwd\_USABaseline}_t = f(\mathbf{X}_t^{USA})$$

Where $\mathbf{X}_t^{USA}$ includes:
- **Economic Indicators**: Unemployment rate, GDP growth, inflation
- **Housing Market Variables**: Existing home sales, mortgage rates, housing starts
- **Financial Conditions**: Federal funds rate, yield curve spreads
- **Lag Features**: Historical HPA values (1, 3, 6, 12, 24 months)
- **Moving Averages**: Rolling averages of key indicators
- **Rate of Change**: Percentage changes in economic variables

**Feature Engineering**:
1. **Lag Features**: $\text{HPA}_{t-1}, \text{HPA}_{t-3}, \text{HPA}_{t-6}, \text{HPA}_{t-12}, \text{HPA}_{t-24}$
2. **Moving Averages**: $\text{MA}_k(\text{HPA})_t = \frac{1}{k}\sum_{i=0}^{k-1} \text{HPA}_{t-i}$
3. **Rate of Change**: $\text{ROC}_k(\text{X})_t = \frac{\text{X}_t - \text{X}_{t-k}}{\text{X}_{t-k}} \times 100$
4. **Min/Max Values**: $\text{Min}_{12}(\text{HPA})_t, \text{Max}_{12}(\text{HPA})_t$

**Model Ensemble**:
$$\text{Prediction}_t = \frac{1}{N}\sum_{i=1}^{N} \text{Model}_i(\mathbf{X}_t)$$

Where models include Ridge Regression, Random Forest, and XGBoost.

### Step 2: MSA Baseline Projection

**Objective**: Generate MSA-specific HPA projections incorporating USA baseline

**Mathematical Model**:
$$\text{ProjectedHPA1YFwd\_MSABaseline}_t^{MSA} = f(\mathbf{X}_t^{MSA}, \text{ProjectedHPA1YFwd\_USABaseline}_t)$$

Where $\mathbf{X}_t^{MSA}$ includes:
- **Regional Economic Data**: MSA-specific unemployment, income, population
- **Housing Market Variables**: Local HPI, housing supply, affordability metrics
- **USA Baseline Integration**: National projections as additional features
- **Regional Lags**: MSA-specific historical HPA patterns
- **Geographic Features**: Population density, economic diversity

**Regional Feature Engineering**:
1. **MSA-Specific Lags**: $\text{HPA}_{t-k}^{MSA}$ for various $k$ values
2. **USA-MSA Deviations**: $\text{HPA}_t^{MSA} - \text{HPA}_t^{USA}$
3. **Regional Moving Averages**: $\text{MA}_k(\text{HPA}^{MSA})_t$
4. **Economic Ratios**: Local unemployment / National unemployment
5. **Housing Affordability**: Median income / Median home price

**Model Architecture**:
- **Individual Models**: Ridge, Random Forest, XGBoost per MSA
- **Ensemble Method**: Simple averaging of base model predictions
- **Time Series Validation**: Forward chaining cross-validation
- **Feature Selection**: VIF analysis and correlation filtering

### Step 3: MSA Calibration

**Objective**: Fine-tune MSA projections using latest economic data

**Mathematical Model**:
$$\text{ProjectedHPA1YFwd\_MSACalibrated}_t^{MSA} = f(\mathbf{X}_t^{New}, \text{ProjectedHPA1YFwd\_MSABaseline}_t^{MSA})$$

Where $\mathbf{X}_t^{New}$ includes:
- **Updated Economic Indicators**: Latest unemployment, income, employment data
- **Market Sentiment**: Consumer confidence, business sentiment
- **Supply-Demand Dynamics**: Housing inventory, days on market
- **Financial Conditions**: Local mortgage rates, credit availability

**Calibration Process**:
1. **Baseline Integration**: Use MSA baseline projections as features
2. **New Data Incorporation**: Add latest economic indicators
3. **Model Retraining**: Update models with recent data
4. **Ensemble Prediction**: Combine multiple model outputs

**Calibration Formula**:
$$\text{Calibrated}_t = \alpha \cdot \text{Baseline}_t + (1-\alpha) \cdot \text{NewModel}_t$$

Where $\alpha$ is determined by model performance on recent data.

## 📈 Statistical Methodology

### Time Series Cross-Validation

**Forward Chaining Approach**:
- Training windows: Expanding window from start to $t-k$
- Validation windows: Fixed-size window at $t-k+1$ to $t$
- Prevents data leakage and mimics real-world forecasting

### Feature Selection and Regularization

**Variance Inflation Factor (VIF)**:
$$\text{VIF}_j = \frac{1}{1-R_j^2}$$

Where $R_j^2$ is the coefficient of determination when feature $j$ is regressed on other features.

**Ridge Regression Regularization**:
$$\min_{\beta} \left\{ \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T\beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}$$

### Model Performance Metrics

**R-squared (Coefficient of Determination)**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

**Mean Absolute Error (MAE)**:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE)**:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 1: Data Preparation

1. **Get FRED API Key**: Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. **Configure System**: Update `config.yaml` with your API key and paths
3. **Run Data Pipeline**: Execute the FRED data collection and processing

```bash
python run_pipeline.py
```

**Enhanced Pipeline**: The pipeline now includes state-level data enhancement:
- **National Data**: Traditional FRED series (unemployment, HPI, etc.)
- **State-Level Data**: 51 states with 15+ economic indicators each
- **Integration**: Combined national and state-level datasets

For state-level data only:
```bash
python batch_state_scraper.py
```

For testing the state-level enhancement:
```bash
python test_state_enhancement.py
```

### Step 2: USA Baseline Projection

```bash
cd USABaseline
python usaBaseline_oneShot.py
```

**Input**: `unified_monthly_data.csv` (processed FRED data)
**Output**: `USA_Baseline_YYYYMMDD.csv` with national HPA projections

### Step 3: MSA Baseline Projection

```bash
python msaBaseline_oneShot.py
```

**Input**: 
- MSA raw data (regional economic indicators)
- USA baseline projections from Step 2

**Output**: `MSA_Baseline_Results.csv` with regional HPA projections

### Step 4: MSA Calibration

```bash
python msaCalibration_oneShot.py
```

**Input**:
- MSA baseline results from Step 3
- Latest MSA economic data

**Output**: `MSA_Calibration_Results.csv` with calibrated projections

## 📊 Output Format

### Main Pipeline Output

The final output contains the following key columns:

| Column | Description | Mathematical Definition |
|--------|-------------|-------------------------|
| `Year_Month_Day` | Date (YYYY-MM-01 format) | Time index $t$ |
| `rcode` | MSA region code | Geographic identifier |
| `cs_name` | MSA region name | Geographic identifier |
| `tag` | Train/test indicator | Data split for validation |
| `ProjectedHPA1YFwd_USABaseline` | USA baseline projections | $f(\mathbf{X}_t^{USA})$ |
| `ProjectedHPA1YFwd_MSABaseline` | MSA baseline projections | $f(\mathbf{X}_t^{MSA}, \text{USA\_Baseline}_t)$ |
| `ProjectedHPA1YFwd_MSACalibrated` | Calibrated projections | $f(\mathbf{X}_t^{New}, \text{MSA\_Baseline}_t)$ |
| `HPI` | Actual HPI values | $\text{HPI}_t$ |
| `hpa12m` | Actual 12-month HPA | $\text{HPA}_{t-12,t}$ |
| `HPA1Yfwd` | 1-year forward HPA (target) | $\text{HPA}_{t,t+12}$ |
| `HPI1Y_fwd` | 1-year forward HPI | $\text{HPI}_{t+12}$ |

### State-Level Data Output

The state-level enhancement produces additional datasets:

**State-Level Data** (`state_level_data.csv`):
```csv
Date,State,SeriesID,Value,Frequency,Source,ProxyFor
2023-01-01,CA,CASTHPI,350.2,monthly,FRED,
2023-01-01,CA,STCAURN,4.1,monthly,FRED,
2023-01-01,CA,CABPPRIVSA,1250,monthly,FRED,TotalShipmentsofNewHomes
```

**Integrated Dataset** (`integrated_state_national_data.csv`):
```csv
Date,Region,State,SeriesID,Value,Frequency,Source,ProxyFor
2023-01-01,United States,CA,CASTHPI,350.2,monthly,FRED,
2023-01-01,United States,TX,TXSTHPI,280.5,monthly,FRED,
2023-01-01,United States,NY,NYSTHPI,420.1,monthly,FRED,
```

**Metadata** (`state_data_metadata.json`):
```json
{
  "coverage_percentage": {"CA": 95.2, "TX": 92.8},
  "successful_series": {"CA": ["UnemploymentRate", "HomePriceIndex"]},
  "failed_series": {"CA": ["MedianDaysonMarket"]},
  "processing_timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

### State-Level Data Enhancement

The pipeline now includes comprehensive state-level data enhancement with the following features:

**Coverage**: 51 states/territories with 15+ economic indicators each
**Series Types**: 
- Monthly indicators (unemployment, HPI, home sales)
- Quarterly indicators (housing units, vacancy rates)
- Proxy series (building permits for new home shipments)

**Special Handling**:
- Realtor.com series with 60% coverage threshold
- Quarterly data interpolation (except vacancy metrics)
- Automatic seasonal adjustment for proxy series
- Comprehensive metadata tracking

**Files Required**:
- `state_level_fred_mappings.txt`: Series ID patterns for each state
- `stateDataEnhancer.py`: Main processing engine
- `batch_state_scraper.py`: Automated processing pipeline

For detailed documentation, see `STATE_LEVEL_README.md`.

### Data Requirements

**USA Data Format**:
```csv
Year_Month_Day,UNRATE,GDP,CPIAUCSL,CSUSHPINSA,...
2020-01-01,3.5,21.4,257.9,250.5,...
2020-02-01,3.6,21.5,258.1,252.1,...
```

**MSA Data Format**:
```csv
Year_Month_Day,rcode,cs_name,HPI,hpa12m,unemployment,income,...
2020-01-01,12345,Metro Area 1,250.5,0.045,3.2,75000,...
2020-02-01,12345,Metro Area 1,252.1,0.047,3.1,75200,...
```

### Model Parameters

**Feature Engineering**:
- Lag periods: [1, 3, 6, 8, 12, 15, 18, 24, 36, 48, 60] months
- Moving averages: [1, 3, 6, 9, 12, 18, 24] months
- Rate of change: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] months

**Model Ensemble**:
- Ridge Regression: $\alpha \in [0.1, 1.0, 10.0]$
- Random Forest: $n\_estimators \in [100, 200]$, $max\_depth \in [10, 20, None]$
- XGBoost: $n\_estimators \in [100, 200]$, $max\_depth \in [6, 10]$, $learning\_rate \in [0.01, 0.1]$

## 📈 Performance Characteristics

### Model Accuracy
- **USA Baseline**: Typical R² = 0.7-0.9
- **MSA Baseline**: Typical R² = 0.6-0.9 (varies by region)
- **MSA Calibration**: Improves accuracy by 5-15% over baseline

### Processing Time
- **USA Baseline**: ~2-5 minutes
- **MSA Baseline**: ~1-5 minutes per MSA region
- **MSA Calibration**: ~1-3 minutes per MSA region

### Memory Usage
- Scales linearly with number of MSAs × time periods
- Recommended: Process in batches for datasets with >100 MSAs

## 🔍 Validation and Testing

### Time Series Validation
- Forward chaining cross-validation prevents data leakage
- Last 12 months reserved for final testing
- Rolling window validation for model stability assessment

### Out-of-Sample Testing
- Compare projected vs actual HPA values
- Calculate prediction intervals for uncertainty quantification
- Monitor model drift and performance degradation

### Regional Validation
- Assess model performance across different MSA types
- Validate economic relationship consistency
- Check for geographic bias in predictions

## 🚨 Error Handling

The pipeline includes robust error handling for:
- Missing data files and insufficient data
- API rate limits and network failures
- Model convergence issues
- Memory constraints for large datasets

MSAs with insufficient data or training failures are automatically skipped with detailed logging.

## 📝 Dependencies

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
requests>=2.25.0
pyyaml>=5.4.0
```

## 🤝 Contributing

To contribute to the pipeline:
1. Follow the existing code structure and error handling patterns
2. Add comprehensive documentation for new features
3. Include unit tests for mathematical functions
4. Validate model performance on sample datasets

## 📄 License

This project is open source. Feel free to modify and distribute according to your needs.

---

**For technical support**: Review the console output for detailed error messages and check the troubleshooting section in individual step documentation. 