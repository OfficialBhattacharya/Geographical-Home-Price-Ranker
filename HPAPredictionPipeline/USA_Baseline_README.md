# USA Baseline Model - Script Documentation

## Overview

The `usaBaseline_oneShot.py` script is a comprehensive time-series forecasting pipeline designed to generate baseline economic projections for the USA. It takes a unified national-level dataset, performs extensive feature engineering, trains an ensemble of machine learning models, and produces 1-year forward projections for a specified target variable.

The script is highly configurable, allowing users to specify input/output paths, date ranges, features, and model parameters.

## Key Features

- **Automated Feature Engineering**: Creates a rich feature set including lags, moving averages, rates of change, and trend deviations.
- **Ensemble Modeling**: Utilizes a `VotingRegressor` that combines predictions from multiple models (e.g., Ridge, Random Forest, XGBoost, LightGBM) to improve forecast accuracy and robustness.
- **Time-Series Cross-Validation**: Employs `TimeSeriesSplit` to ensure that the model evaluation is appropriate for temporal data, preventing lookahead bias.
- **Data Preprocessing**: Includes steps for handling missing values, removing skewness/kurtosis, standardizing data, and managing multicollinearity via VIF analysis.
- **Configurability**: Offers multiple ways to run the pipeline: with default paths, custom paths, or a fully interactive configuration session.
- **Visualization**: Generates plots to visualize model predictions against actual values, feature importance, and prediction intervals.

## Input Data Requirements

The script expects a single CSV file with national-level time-series data.

### Input Data Format (`unified_monthly_data.csv`):
```csv
Date,Feature1,Feature2,Target_HPA,[additional_features...]
1990-01-01,102.5,0.05,0.03, ...
1990-02-01,102.8,0.051,0.032, ...
1990-03-01,103.1,0.049,0.031, ...
...
```

**Required Columns:**
- **Date Column**: A date/time column, specified by the user (default: `Date`).
- **Feature Columns**: Numerical columns used for prediction.
- **Target Column**: The variable to be forecasted (e.g., `HPA12m_USA`).

## Usage

The script can be executed in several ways.

### Method 1: Default Configuration
This runs the pipeline with hardcoded default file paths and parameters.
```python
# Run with default settings
main()
```

### Method 2: Custom File Paths
This allows specifying the input and output file paths directly.
```python
# Run with your specific data files
run_with_custom_paths(
    usa_file="path/to/your/usa_data.csv",
    output_file="path/to/your/output.csv"
)
```

### Method 3: Interactive Configuration
This prompts the user to enter all configurations in the console.
```python
# Get user input for all configurations
cfg = CFG(get_user_input=True)
# ... then run the pipeline using this cfg object
```

## Model Architecture

### 1. Feature Engineering
- **Lags**: Lagged values of the target and feature variables (e.g., 1, 3, 6, 12, 24 months ago).
- **Moving Averages**: Simple moving averages over various windows (e.g., 3, 6, 12 months).
- **Rates of Change**: Percentage change over different periods.
- **Trend Deviation**: Deviation from a rolling average trend.

### 2. Data Preprocessing
- **Missing Value Imputation**: Fills missing data using methods like forward-fill, back-fill, or a decay rate mechanism.
- **Transformations**: Applies Box-Cox or Yeo-Johnson transformations to reduce data skewness.
- **Standardization**: Scales features using `RobustScaler` (default), `StandardScaler`, or `MinMaxScaler`.
- **Collinearity Reduction**: Uses Variance Inflation Factor (VIF) to iteratively remove features with high multicollinearity.

### 3. Modeling Approach
- **Base Models**: A suite of regressors are trained:
    - `LinearRegression`
    - `Ridge`
    - `Lasso`
    - `ElasticNet`
    - `RandomForestRegressor`
    - `GradientBoostingRegressor`
    - `XGBoost`
    - `LightGBM`
- **Ensemble Method**: A `VotingRegressor` combines the predictions from the best-performing base models. The weights can be uniform or based on model performance.
- **Train/Test Split**: The script splits the data chronologically. A `targetForward` parameter (default: 12 months) defines the forecast horizon. Data points without a future target value become the test set for prediction.

### 4. Processing Flow
1. **Load Data**: Loads the data and ensures all months in the specified range are present.
2. **Create Target**: Creates the forward-looking target variable (e.g., `HPA1Yfwd`).
3. **Engineer Features**: Generates the full set of features.
4. **Preprocess Data**: Imputes missing values, transforms skewed data, and scales features.
5. **Select Features**: Removes collinear features using VIF.
6. **Train Models**: Trains each base model using time-series cross-validation.
7. **Create Ensemble**: Builds a `VotingRegressor` from the selected models.
8. **Generate Predictions**: Produces forecasts for the test period.
9. **Compile and Save Output**: Saves the results to a CSV file.

## Output Description

The final output CSV file contains the historical data along with the model's predictions.

- **Date**: The timestamp for each data point.
- **[Original_Features]**: All columns from the input file.
- **[Engineered_Features]**: The features created during the process.
- **[Target_Column]**: The actual future values (where available).
- **Projected[Target_Column]**: The model's 1-year forward predictions.
- **Prediction_Lower_Bound**: The lower bound of the prediction interval.
- **Prediction_Upper_Bound**: The upper bound of the prediction interval.

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
statsmodels
```

## Error Handling

The script includes logging to capture information, warnings, and errors during execution. It is designed to handle:
- Missing data files or incorrect paths.
- Missing date values.
- Failures during model training. 