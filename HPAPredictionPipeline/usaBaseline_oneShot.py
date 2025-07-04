import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import boxcox
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Global variable to store USA baseline results
USA_RESULTS = {}

class CFG:
    """
    General configuration class for data and model setup.
    Takes user inputs for all necessary configurations.
    """
    
    def __init__(self, get_user_input=True):
        logger.info("Initializing CFG class...")
        
        if get_user_input:
            self._get_user_configurations()
        else:
            self._set_default_configurations()
        
        self._setup_plotting_specs()
        logger.info("CFG initialization complete.")
    
    def _get_user_configurations(self):
        """Get all configurations from user input"""
        print("=== CONFIGURATION SETUP ===")
        
        # Data paths
        self.filePath = input("Enter input CSV file path [D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv]: ") or "D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv"
        
        # Date configurations
        self.dateCol = input("Enter date column name [Date]: ") or "Date"
        self.start_date = input("Enter start date (YYYY-MM-DD) [1990-01-01]: ") or "1990-01-01"
        self.end_date = input("Enter end date (YYYY-MM-DD): ")
        if not self.end_date:
            raise ValueError("End date is required. Please provide a valid date in YYYY-MM-DD format.")
        
        # Generate output path with end date
        end_date_formatted = self.end_date.replace('-', '')
        self.outputPath = input(f"Enter output CSV file path [D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_{end_date_formatted}.csv]: ") or f"D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_{end_date_formatted}.csv"
        
        # Feature configurations
        feature_input = input("Enter feature columns (comma-separated) []: ")
        self.featureList = [f.strip() for f in feature_input.split(",")] if feature_input else []
        
        # Target and HPI configurations
        target_input = input("Enter target column name [hpa12m]: ")
        self.targetCol = target_input if target_input else "hpa12m"
        
        hpi_input = input("Enter HPI column name [HPI]: ")
        self.hpiCol = hpi_input if hpi_input else "HPI"
        
        # ID columns
        id_input = input("Enter ID columns (comma-separated) []: ")
        self.idList = [i.strip() for i in id_input.split(",")] if id_input else []
        
        # Lag configurations
        lag_input = input("Enter lag values (comma-separated) [1,3,6,8,12,15,18,24,36,48,60]: ")
        self.lagList = [int(l.strip()) for l in lag_input.split(",")] if lag_input else [1,3,6,8,12,15,18,24,36,48,60]
        
        # Rate configurations
        rate_input = input("Enter rate values (comma-separated) [1,2,3,4,5,6,7,8,9,10,11,12]: ")
        self.rateList = [int(r.strip()) for r in rate_input.split(",")] if rate_input else [1,2,3,4,5,6,7,8,9,10,11,12]
        
        # Moving average configurations
        ma_input = input("Enter moving average periods (comma-separated) [1,3,6,9,12,18,24]: ")
        self.movingAverages = [int(m.strip()) for m in ma_input.split(",")] if ma_input else [1,3,6,9,12,18,24]
        
        # Target forward
        target_forward = input("Enter target forward months [12]: ")
        self.targetForward = int(target_forward) if target_forward else 12
        
        # Features to force use
        forced_features = input("Enter features to force use (comma-separated) []: ")
        self.featuresToUse = [f.strip() for f in forced_features.split(",")] if forced_features else []
        
        # Model configurations
        self._setup_model_configurations()
    
    def _set_default_configurations(self):
        """Set default configurations"""
        self.filePath = "D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv"
        self.outputPath = None  # Will be set based on end_date
        self.idList = []
        self.dateCol = "Date"
        self.start_date = "1990-01-01"
        self.end_date = None  # Must be provided by user
        self.featureList = []
        self.targetCol = "hpa12m"
        self.hpiCol = "HPI"
        self.lagList = [1,3,6,8,12,15,18,24,36,48,60]
        self.rateList = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.movingAverages = [1,3,6,9,12,18,24]
        self.targetForward = 12
        self.featuresToUse = []
        self._setup_model_configurations()
    
    def _setup_model_configurations(self):
        """Setup model configurations"""
        self.AllModelsList = [
            'LinearRegression',
            'Ridge',
            'Lasso',
            'ElasticNet',
            'RandomForest',
            'XGBoost',
            'LightGBM',
            'GradientBoosting'
        ]
        
        self.AllModelParams = {
            'LinearRegression': {},
            'Ridge': {'alpha': [0.1, 1.0, 10.0]},
            'Lasso': {'alpha': [0.1, 1.0, 10.0]},
            'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
            'LightGBM': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
            'GradientBoosting': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
        }
    
    def _setup_plotting_specs(self):
        """Setup plotting specifications"""
        self.grid_specs = {
            'visible': True,
            'which': 'both',
            'linestyle': '--',
            'color': 'lightgrey',
            'linewidth': 0.75
        }
        
        self.title_specs = {
            'fontsize': 9,
            'fontweight': 'bold',
            'color': '#992600',
        }


def loadDataAndCheckAllMonths(filePath, date_col, start_date, end_date, date_format="%Y-%m-%d"):
    """
    Load data and check if all months have rows from start to end date.
    
    Parameters:
    - filePath: Path to the CSV file
    - date_col: Name of the date column
    - start_date: Start date string
    - end_date: End date string
    - date_format: Date format string
    
    Returns:
    - df: Filtered dataframe
    """
    logger.info("Step 2: Loading data and checking for all months...")
    print(f"Loading data from: {filePath}")
    
    try:
        # Load data
        df = pd.read_csv(filePath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Original data shape: {df.shape}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        
        # Filter data by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df_filtered = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
        logger.info(f"Data filtered to date range {start_date} to {end_date}. Shape: {df_filtered.shape}")
        print(f"Filtered data shape: {df_filtered.shape}")
        
        # Check for missing months
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # Month start frequency
        existing_dates = df_filtered[date_col].dt.to_period('M').unique()
        expected_dates = date_range.to_period('M')
        
        missing_months = set(expected_dates) - set(existing_dates)
        
        if missing_months:
            logger.warning(f"Missing months detected: {sorted(missing_months)}")
            print(f"WARNING: Missing {len(missing_months)} months: {sorted(missing_months)}")
        else:
            logger.info("All months present in the specified date range.")
            print("✓ All months present in the specified date range.")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Error in loadDataAndCheckAllMonths: {str(e)}")
        raise e


def addAllFeatures(df, idList, dateCol, featureList, targetCol, lagList, movingAverages, rateList):
    """
    Add all lagged features, moving averages, log differences, min/max values,
    trendlines, deviations, and growth rates.
    
    Parameters:
    - df: Input dataframe
    - idList: List of ID columns
    - dateCol: Date column name
    - featureList: List of feature columns
    - targetCol: Target column name
    - lagList: List of lag periods
    - movingAverages: List of moving average periods
    - rateList: List of rate periods
    
    Returns:
    - df: Enhanced dataframe with new features
    """
    logger.info("Step 3: Adding all features...")
    print("Adding comprehensive feature set...")
    
    df_enhanced = df.copy()
    
    # Combine features and target for processing
    all_features = featureList + [targetCol] if targetCol and targetCol not in featureList else featureList
    
    for feature in all_features:
        if feature not in df_enhanced.columns:
            logger.warning(f"Feature {feature} not found in dataframe")
            continue
            
        print(f"Processing feature: {feature}")
        
        # Sort by date for proper lag calculation
        df_enhanced = df_enhanced.sort_values([dateCol])
        
        # 1. Lagged features
        for lag in lagList:
            lag_col = f"{feature}_lag_{lag}"
            df_enhanced[lag_col] = df_enhanced.groupby(idList if idList else [True])[feature].shift(lag)
            
        # 2. Moving averages
        for ma in movingAverages:
            ma_col = f"{feature}_ma_{ma}"
            df_enhanced[ma_col] = df_enhanced.groupby(idList if idList else [True])[feature].rolling(window=ma, min_periods=1).mean().reset_index(level=0, drop=True)
            
        # 3. Log differences
        for lag in lagList:
            log_diff_col = f"{feature}_log_diff_{lag}"
            df_enhanced[log_diff_col] = df_enhanced.groupby(idList if idList else [True])[feature].apply(
                lambda x: np.log(x / x.shift(lag)).replace([np.inf, -np.inf], np.nan)
            ).reset_index(level=0, drop=True)
            
        # 4. Min/Max in last year (12 months)
        min_col = f"{feature}_min_12m"
        max_col = f"{feature}_max_12m"
        df_enhanced[min_col] = df_enhanced.groupby(idList if idList else [True])[feature].rolling(window=12, min_periods=1).min().reset_index(level=0, drop=True)
        df_enhanced[max_col] = df_enhanced.groupby(idList if idList else [True])[feature].rolling(window=12, min_periods=1).max().reset_index(level=0, drop=True)
        
        # 5. Trendlines and deviations (using linear regression on rolling window)
        def calculate_trend_deviation(series, window=12):
            if len(series) < window:
                return pd.Series([np.nan, np.nan], index=['trend', 'deviation'])
            
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            trend = slope * (len(series) - 1) + intercept
            deviation = series.iloc[-1] - trend
            
            return pd.Series([slope, deviation], index=['trend', 'deviation'])
        
        trend_dev = df_enhanced.groupby(idList if idList else [True])[feature].rolling(window=12, min_periods=6).apply(
            lambda x: calculate_trend_deviation(x).iloc[0], raw=False
        ).reset_index(level=0, drop=True)
        
        df_enhanced[f"{feature}_trend_12m"] = trend_dev
        
        deviation = df_enhanced.groupby(idList if idList else [True])[feature].rolling(window=12, min_periods=6).apply(
            lambda x: calculate_trend_deviation(x).iloc[1], raw=False
        ).reset_index(level=0, drop=True)
        
        df_enhanced[f"{feature}_deviation_12m"] = deviation
        
        # 6. Growth rates
        for rate in rateList:
            growth_col = f"{feature}_growth_{rate}m"
            df_enhanced[growth_col] = df_enhanced.groupby(idList if idList else [True])[feature].pct_change(periods=rate)
    
    logger.info(f"Feature engineering complete. New shape: {df_enhanced.shape}")
    print(f"Feature engineering complete. DataFrame shape: {df_enhanced.shape}")
    
    return df_enhanced

def addTarget(df, idList, dateCol, targetCol, hpiCol, targetForward):
    """
    Create forward-looking target variable and HPI forward variable.
    
    Parameters:
    - df: Input dataframe
    - idList: List of ID columns
    - dateCol: Date column name
    - targetCol: Original target column name
    - hpiCol: HPI column name
    - targetForward: Number of months to forward
    
    Returns:
    - df: Dataframe with new target and HPI forward columns
    """
    logger.info("Step 4: Adding target and HPI forward variables...")
    print(f"Creating target variable with {targetForward} months forward look...")
    
    df_target = df.copy()
    
    # Sort by date for proper forward calculation
    df_target = df_target.sort_values([dateCol])
    
    # Create forward target (12-month forward of hpa12m)
    new_target_col = f"{targetCol}_forward_{targetForward}m"
    df_target[new_target_col] = df_target.groupby(idList if idList else [True])[targetCol].shift(-targetForward)
    
    # Create forward HPI (12-month forward of HPI)
    new_hpi_col = f"{hpiCol}_forward_{targetForward}m"
    df_target[new_hpi_col] = df_target.groupby(idList if idList else [True])[hpiCol].shift(-targetForward)
    
    logger.info(f"Target variable '{new_target_col}' created successfully")
    logger.info(f"HPI forward variable '{new_hpi_col}' created successfully")
    print(f"✓ Target variable '{new_target_col}' created")
    print(f"✓ HPI forward variable '{new_hpi_col}' created")
    
    return df_target, new_target_col, new_hpi_col

def fillMissingValues(df, new_target_col, idList, dateCol, fillMethod="DecayRate"):
    """
    Fill missing values using decay rate backfill. Identify train/test split.
    
    Parameters:
    - df: Input dataframe
    - new_target_col: Name of the new target column
    - idList: List of ID columns
    - dateCol: Date column name
    - fillMethod: Method for filling missing values
    
    Returns:
    - df_clean: Cleaned dataframe
    - train_df: Training data
    - test_df: Test data
    """
    logger.info("Step 5: Filling missing values and creating train/test split...")
    print("Filling missing values and identifying train/test datasets...")
    
    df_filled = df.copy()
    
    # Identify X variables (all columns except ID, date, and target)
    exclude_cols = idList + [dateCol, new_target_col]
    all_columns = [col for col in df_filled.columns if col not in exclude_cols]
    
    # Filter to only include numeric columns
    x_columns = []
    non_numeric_columns = []
    
    for col in all_columns:
        try:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                x_columns.append(col)
            else:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df_filled[col], errors='coerce')
                if numeric_col.isna().sum() <= len(numeric_col) * 0.5:  # Less than 50% NaN
                    df_filled[col] = numeric_col
                    x_columns.append(col)
                else:
                    non_numeric_columns.append(col)
        except Exception as e:
            non_numeric_columns.append(col)
            logger.warning(f"Column {col} could not be converted to numeric: {str(e)}")
    
    print(f"Number of X variables: {len(x_columns)}")
    if non_numeric_columns:
        print(f"Excluded non-numeric columns: {non_numeric_columns}")
        logger.info(f"Excluded {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns}")
    
    # Fill missing values in X variables using decay rate (exponential weighted mean)
    for col in x_columns:
        if df_filled[col].isnull().any():
            # Group by ID if ID columns exist
            if idList:
                df_filled[col] = df_filled.groupby(idList)[col].apply(
                    lambda x: x.fillna(method='bfill').fillna(method='ffill')
                ).reset_index(level=0, drop=True)
            else:
                # Use exponential weighted mean for backfill
                df_filled[col] = df_filled[col].fillna(method='bfill').fillna(method='ffill')
                
                # Apply decay rate
                if df_filled[col].isnull().any():
                    ewm_values = df_filled[col].ewm(span=12, adjust=False).mean()
                    df_filled[col] = df_filled[col].fillna(ewm_values)
    
    # Create train/test split based on target availability
    train_mask = df_filled[new_target_col].notna()
    test_mask = df_filled[new_target_col].isna()
    
    train_df = df_filled[train_mask].copy()
    test_df = df_filled[test_mask].copy()
    
    # Verify no NAs in X variables for training data
    train_x_nas = train_df[x_columns].isnull().sum().sum()
    if train_x_nas > 0:
        logger.warning(f"Found {train_x_nas} NAs in training X variables")
        # Final cleanup for training data
        for col in x_columns:
            if train_df[col].isnull().any():
                train_df[col] = train_df[col].fillna(train_df[col].median())
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    print(f"✓ Training data: {train_df.shape[0]} rows")
    print(f"✓ Test data: {test_df.shape[0]} rows")
    
    return df_filled, train_df, test_df, x_columns

def removeSkewnessAndKurtosis(train_df, test_df, x_columns, threshold=1):
    """
    Remove skewness and kurtosis using Box-Cox transformation.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe
    - x_columns: List of X variable columns
    - threshold: Skewness threshold for transformation
    
    Returns:
    - train_transformed: Transformed training data
    - test_transformed: Transformed test data
    - transformers: Dictionary of transformers used
    """
    logger.info("Step 6: Removing skewness and kurtosis...")
    print("Applying Box-Cox transformations to reduce skewness...")
    
    train_transformed = train_df.copy()
    test_transformed = test_df.copy()
    transformers = {}
    
    for col in x_columns:
        try:
            # Check if column contains numeric data
            if not pd.api.types.is_numeric_dtype(train_transformed[col]):
                logger.warning(f"Column {col} is not numeric (dtype: {train_transformed[col].dtype}), skipping transformation")
                transformers[col] = None
                continue
            
            # Check for any non-numeric values that might have been missed
            try:
                # Try to convert to numeric, coercing errors to NaN
                train_numeric = pd.to_numeric(train_transformed[col], errors='coerce')
                test_numeric = pd.to_numeric(test_transformed[col], errors='coerce')
                
                # Check if we have enough valid numeric values
                if train_numeric.isna().sum() > len(train_numeric) * 0.5:  # More than 50% NaN
                    logger.warning(f"Column {col} has too many non-numeric values, skipping transformation")
                    transformers[col] = None
                    continue
                
                # Update the columns with cleaned numeric data
                train_transformed[col] = train_numeric
                test_transformed[col] = test_numeric
                
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                transformers[col] = None
                continue
            
            # Calculate skewness
            skewness = abs(train_transformed[col].skew())
            
            if skewness > threshold:
                print(f"Transforming {col} (skewness: {skewness:.3f})")
                
                # Ensure positive values for Box-Cox
                min_val = train_transformed[col].min()
                if min_val <= 0:
                    shift_value = abs(min_val) + 1
                    train_transformed[col] = train_transformed[col] + shift_value
                    test_transformed[col] = test_transformed[col] + shift_value
                    transformers[col] = {'shift': shift_value}
                else:
                    transformers[col] = {'shift': 0}
                
                # Apply Box-Cox transformation
                transformed_values, lambda_param = boxcox(train_transformed[col])
                train_transformed[col] = transformed_values
                transformers[col]['lambda'] = lambda_param
                
                # Apply same transformation to test data
                if lambda_param == 0:
                    test_transformed[col] = np.log(test_transformed[col])
                else:
                    test_transformed[col] = (np.power(test_transformed[col], lambda_param) - 1) / lambda_param
                
                new_skewness = abs(pd.Series(transformed_values).skew())
                logger.info(f"Column {col}: skewness reduced from {skewness:.3f} to {new_skewness:.3f}")
                
        except Exception as e:
            logger.warning(f"Could not transform column {col}: {str(e)}")
            transformers[col] = None
    
    logger.info("Skewness and kurtosis reduction complete")
    print("✓ Skewness and kurtosis reduction complete")
    
    return train_transformed, test_transformed, transformers

def standardizeData(train_df, test_df, x_columns, scaler_type="RobustScaler"):
    """
    Standardize features using specified scaler.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe
    - x_columns: List of X variable columns
    - scaler_type: Type of scaler ('RobustScaler', 'StandardScaler', 'MinMaxScaler')
    
    Returns:
    - train_scaled: Scaled training data
    - test_scaled: Scaled test data
    - scaler: Fitted scaler object
    """
    logger.info("Step 7: Standardizing data...")
    print(f"Standardizing data using {scaler_type}...")
    
    # Choose scaler
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        logger.warning(f"Unknown scaler type {scaler_type}, using RobustScaler")
        scaler = RobustScaler()
    
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    # Ensure all columns are numeric before scaling
    numeric_x_columns = []
    for col in x_columns:
        if pd.api.types.is_numeric_dtype(train_scaled[col]):
            numeric_x_columns.append(col)
        else:
            logger.warning(f"Column {col} is not numeric, skipping standardization")
    
    if not numeric_x_columns:
        logger.error("No numeric columns found for standardization")
        raise ValueError("No numeric columns found for standardization")
    
    # Fit scaler on training data and transform both
    scaler.fit(train_scaled[numeric_x_columns])
    
    train_scaled[numeric_x_columns] = scaler.transform(train_scaled[numeric_x_columns])
    test_scaled[numeric_x_columns] = scaler.transform(test_scaled[numeric_x_columns])
    
    logger.info("Data standardization complete")
    print("✓ Data standardization complete")
    
    return train_scaled, test_scaled, scaler, numeric_x_columns

def checkAndRemoveHighVIF(train_df, test_df, x_columns, threshold=10, min_features=10, fallback_features=15):
    """
    Efficiently check for and remove high VIF features with non-iterative approach.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe  
    - x_columns: List of X variable columns
    - threshold: VIF threshold for removal (default: 10)
    - min_features: Minimum features to keep if possible (default: 10)
    - fallback_features: Number of lowest VIF features to keep as fallback (default: 15)
    
    Returns:
    - train_clean: Training data with low VIF features
    - test_clean: Test data with low VIF features
    - final_features: List of remaining features
    """
    logger.info("Step 8: Checking and removing high VIF features (optimized approach)...")
    print(f"Calculating VIF for {len(x_columns)} features...")
    
    start_time = pd.Timestamp.now()
    
    try:
        # Calculate VIF for all features at once
        X_temp = train_df[x_columns].values
        
        # Handle any remaining NaN or infinite values
        if np.any(np.isnan(X_temp)) or np.any(np.isinf(X_temp)):
            logger.warning("Found NaN or infinite values in features, cleaning...")
            X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = x_columns
        
        # Calculate VIF for all features
        print("Computing VIF values...")
        vif_values = []
        for i in range(len(x_columns)):
            try:
                vif = variance_inflation_factor(X_temp, i)
                # Handle infinite or very large VIF values
                if np.isnan(vif) or np.isinf(vif):
                    vif = 999999  # Set to very high value for removal
                vif_values.append(vif)
            except Exception as e:
                logger.warning(f"Error calculating VIF for feature {x_columns[i]}: {str(e)}")
                vif_values.append(999999)  # Set to very high value for removal
        
        vif_data["VIF"] = vif_values
        
        # Sort by VIF (ascending order - lowest VIF first)
        vif_data = vif_data.sort_values('VIF').reset_index(drop=True)
        
        print(f"VIF calculation complete. Range: {vif_data['VIF'].min():.2f} to {vif_data['VIF'].max():.2f}")
        
        # Strategy 1: Remove all features with VIF > threshold
        low_vif_features = vif_data[vif_data['VIF'] <= threshold]['Feature'].tolist()
        high_vif_features = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
        
        print(f"Features with VIF <= {threshold}: {len(low_vif_features)}")
        print(f"Features with VIF > {threshold}: {len(high_vif_features)}")
        
        # Apply the logic: keep low VIF if >= min_features, otherwise keep top fallback_features
        if len(low_vif_features) >= min_features:
            final_features = low_vif_features
            removed_features = high_vif_features
            strategy_used = f"Kept {len(final_features)} features with VIF <= {threshold}"
        else:
            # Fallback: keep the features with lowest VIF values
            final_features = vif_data.head(fallback_features)['Feature'].tolist()
            removed_features = [f for f in x_columns if f not in final_features]
            strategy_used = f"Kept top {fallback_features} features with lowest VIF (fallback strategy)"
        
        # Create clean datasets
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        # Log results
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        logger.info(f"VIF filtering complete in {execution_time:.2f} seconds")
        logger.info(f"Strategy used: {strategy_used}")
        logger.info(f"Removed {len(removed_features)} features due to high VIF")
        logger.info(f"Final feature count: {len(final_features)}")
        
        print(f"✓ VIF filtering complete in {execution_time:.2f} seconds")
        print(f"✓ Strategy: {strategy_used}")
        print(f"✓ Removed {len(removed_features)} high VIF features")
        print(f"✓ Final feature count: {len(final_features)}")
        
        # Show some statistics
        if len(final_features) > 0:
            final_vif_stats = vif_data[vif_data['Feature'].isin(final_features)]['VIF']
            print(f"✓ Final VIF range: {final_vif_stats.min():.2f} to {final_vif_stats.max():.2f}")
            print(f"✓ Mean VIF of remaining features: {final_vif_stats.mean():.2f}")
        
        # Log top removed features for reference
        if len(removed_features) > 0:
            removed_vif_data = vif_data[vif_data['Feature'].isin(removed_features)].head(10)
            logger.info("Top 10 removed features by VIF:")
            for _, row in removed_vif_data.iterrows():
                logger.info(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")
        
        return train_clean, test_clean, final_features
        
    except Exception as e:
        logger.error(f"Error in VIF calculation: {str(e)}")
        logger.warning("Falling back to original feature set due to VIF calculation error")
        print(f"❌ VIF calculation failed: {str(e)}")
        print("⚠️  Using all original features as fallback")
        
        # Return original features as fallback
        return train_df.copy(), test_df.copy(), x_columns

def timeseriesCV(scheme_type="TimeSeriesSplit", n_splits=5, test_size=None):
    """
    Create time series cross-validation scheme.
    
    Parameters:
    - scheme_type: Type of CV scheme
    - n_splits: Number of splits
    - test_size: Test size for each split
    
    Returns:
    - cv_scheme: Cross-validation object
    """
    logger.info("Step 9: Setting up time series cross-validation...")
    print(f"Setting up {scheme_type} with {n_splits} splits...")
    
    if scheme_type == "TimeSeriesSplit":
        cv_scheme = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    else:
        logger.warning(f"Unknown CV scheme {scheme_type}, using TimeSeriesSplit")
        cv_scheme = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    logger.info("Time series CV setup complete")
    print("✓ Time series CV setup complete")
    
    return cv_scheme

def fitAndPredictVotingRegressor(train_df, test_df, final_features, featuresToUse, new_target_col, 
                                modelList, modelParams, cv_scheme, scaler=None, date_col='Date'):
    """
    Fit multiple models and create neural network-based meta learner for final predictions.
    
    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe
    - final_features: List of final features
    - featuresToUse: Features to force include
    - new_target_col: Target column name
    - modelList: List of model names
    - modelParams: Model parameters dictionary
    - cv_scheme: Cross-validation scheme
    - scaler: Fitted scaler object
    - date_col: Date column name for time-series plotting
    
    Returns:
    - results: Dictionary containing model results and predictions
    """
    logger.info("Step 10: Training models and creating neural network meta learner...")
    print("Training multiple models and creating neural network ensemble...")
    
    # Combine final features with forced features
    all_features = list(set(final_features + featuresToUse))
    
    # Handle forced features that might need preprocessing
    for feature in featuresToUse:
        if feature not in train_df.columns:
            logger.warning(f"Forced feature {feature} not found in data")
            continue
        
        if feature not in final_features:
            # Fill missing values and standardize forced features
            if train_df[feature].isnull().any():
                train_df[feature] = train_df[feature].fillna(train_df[feature].median())
                test_df[feature] = test_df[feature].fillna(test_df[feature].median())
            
            if scaler is not None:
                # Standardize forced features
                feature_scaler = type(scaler)()
                train_df[feature] = feature_scaler.fit_transform(train_df[[feature]])
                test_df[feature] = feature_scaler.transform(test_df[[feature]])
    
    # Prepare data
    X_train = train_df[all_features]
    y_train = train_df[new_target_col]
    X_test = test_df[all_features]
    
    # Initialize models with improved hyperparameters
    models = {}
    
    if 'LinearRegression' in modelList:
        models['LinearRegression'] = LinearRegression()
    
    if 'Ridge' in modelList:
        # Grid search for best Ridge alpha
        ridge_params = modelParams.get('Ridge', {'alpha': [0.1, 1.0, 10.0, 100.0]})
        ridge = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['Ridge'] = ridge
    
    if 'Lasso' in modelList:
        # Grid search for best Lasso alpha
        lasso_params = modelParams.get('Lasso', {'alpha': [0.01, 0.1, 1.0, 10.0]})
        lasso = GridSearchCV(Lasso(max_iter=2000), lasso_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['Lasso'] = lasso
    
    if 'ElasticNet' in modelList:
        # Grid search for best ElasticNet parameters
        elastic_params = modelParams.get('ElasticNet', {
            'alpha': [0.01, 0.1, 1.0, 10.0], 
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        })
        elastic = GridSearchCV(ElasticNet(max_iter=2000), elastic_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['ElasticNet'] = elastic
    
    if 'RandomForest' in modelList:
        # Improved RandomForest parameters
        rf_params = modelParams.get('RandomForest', {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        })
        rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, 
                         cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['RandomForest'] = rf
    
    if 'XGBoost' in modelList:
        # Improved XGBoost parameters
        xgb_params = modelParams.get('XGBoost', {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        })
        xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_params,
                                cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['XGBoost'] = xgb_model
    
    if 'LightGBM' in modelList:
        # Improved LightGBM parameters
        lgb_params = modelParams.get('LightGBM', {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100, 150],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        })
        lgb_model = GridSearchCV(lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1), lgb_params,
                                cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['LightGBM'] = lgb_model
    
    if 'GradientBoosting' in modelList:
        # Improved GradientBoosting parameters
        gb_params = modelParams.get('GradientBoosting', {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        })
        gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params,
                               cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        models['GradientBoosting'] = gb_model
    
    # Train individual models and collect predictions for meta learner
    model_scores = {}
    trained_models = {}
    base_predictions_train = []
    base_predictions_test = []
    
    print("Training individual models with hyperparameter tuning...")
    
    for name, model in models.items():
        print(f"Training and tuning {name}...")
        
        try:
            # Fit model (with grid search if applicable)
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Get predictions for meta learner
            if hasattr(model, 'predict'):
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            else:
                # For GridSearchCV objects
                train_pred = model.best_estimator_.predict(X_train)
                test_pred = model.best_estimator_.predict(X_test)
            
            base_predictions_train.append(train_pred)
            base_predictions_test.append(test_pred)
            
            # Calculate CV score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_scheme, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            model_scores[name] = -cv_scores.mean()
            
            logger.info(f"{name} - CV MSE: {model_scores[name]:.4f}")
            
            # Log best parameters if it's a GridSearchCV
            if hasattr(model, 'best_params_'):
                logger.info(f"{name} best parameters: {model.best_params_}")
                print(f"  Best params: {model.best_params_}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            print(f"  ❌ Error training {name}: {str(e)}")
    
    if len(base_predictions_train) < 2:
        logger.error("Not enough models trained successfully for ensemble")
        raise ValueError("Not enough models trained successfully for ensemble")
    
    # Prepare meta learner data
    meta_X_train = np.column_stack(base_predictions_train)
    meta_X_test = np.column_stack(base_predictions_test)
    
    print(f"Training neural network meta learner on {meta_X_train.shape[1]} base model predictions...")
    
    # Create neural network meta learner with regularization
    meta_learner = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2 regularization to prevent overfitting to one model
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    )
    
    # Train meta learner
    meta_learner.fit(meta_X_train, y_train)
    
    # Make final predictions
    final_train_pred = meta_learner.predict(meta_X_train)
    final_test_pred = meta_learner.predict(meta_X_test)
    
    # Calculate final metrics
    train_mse = mean_squared_error(y_train, final_train_pred)
    train_r2 = r2_score(y_train, final_train_pred)
    train_mae = mean_absolute_error(y_train, final_train_pred)
    
    logger.info(f"Neural Network Meta Learner - Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"✓ Neural Network Meta Learner trained - Train R²: {train_r2:.4f}")
    
    # Calculate approximate model weights (for interpretation)
    try:
        # Use a simple method to estimate model importance
        model_weights = {}
        model_names = list(trained_models.keys())
        
        # Calculate correlation between base predictions and meta learner output
        correlations = []
        for i, pred in enumerate(base_predictions_train):
            corr = np.corrcoef(pred, final_train_pred)[0, 1]
            correlations.append(abs(corr))
        
        # Normalize to get approximate weights
        total_corr = sum(correlations)
        if total_corr > 0:
            for i, name in enumerate(model_names):
                model_weights[name] = correlations[i] / total_corr
        else:
            # Equal weights as fallback
            for name in model_names:
                model_weights[name] = 1.0 / len(model_names)
        
        # Check if any model has >70% weight and log warning
        max_weight = max(model_weights.values())
        max_weight_model = max(model_weights, key=model_weights.get)
        
        if max_weight > 0.7:
            logger.warning(f"Model {max_weight_model} has high weight ({max_weight:.3f}). Consider increasing regularization.")
            print(f"⚠️  Warning: {max_weight_model} has high weight ({max_weight:.3f})")
        
        logger.info(f"Approximate model weights: {model_weights}")
        print(f"📊 Model weights: {model_weights}")
        
    except Exception as e:
        logger.warning(f"Could not calculate model weights: {str(e)}")
        model_weights = {}
    
    # Compile results with enhanced information
    results = {
        'meta_learner': meta_learner,
        'individual_models': trained_models,
        'model_scores': model_scores,
        'model_weights': model_weights,
        'train_predictions': final_train_pred,
        'test_predictions': final_test_pred,
        'base_train_predictions': dict(zip(trained_models.keys(), base_predictions_train)),
        'base_test_predictions': dict(zip(trained_models.keys(), base_predictions_test)),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'features_used': all_features,
        'target_column': new_target_col,
        # Add date information for time-series plotting
        'train_dates': train_df[date_col] if date_col in train_df.columns else None,
        'test_dates': test_df[date_col] if date_col in test_df.columns else None,
        'date_column': date_col,
        # Meta learner specific info
        'meta_train_data': meta_X_train,
        'meta_test_data': meta_X_test
    }
    
    logger.info("Model training and neural network ensemble creation complete")
    print("✓ Model training and neural network ensemble creation complete")
    
    return results

def plotResults(results, output_path=None):
    """
    Plot actual vs predicted values for the final model including time-series analysis.
    
    Parameters:
    - results: Results dictionary from fitAndPredictVotingRegressor
    - output_path: Path to save plots
    
    Returns:
    - None (displays/saves plots)
    """
    logger.info("Step 11: Plotting results...")
    print("Creating comprehensive model analysis plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time-Series Plot: Actual vs Predicted over Time
    ax1 = axes[0]
    
    if results['train_dates'] is not None and results['test_dates'] is not None:
        # Plot training period (actual vs predicted)
        train_dates = pd.to_datetime(results['train_dates'])
        test_dates = pd.to_datetime(results['test_dates'])
        
        # Plot actual values for training period
        ax1.plot(train_dates, results['y_train'], 'b-', label='Actual (Training)', linewidth=2, alpha=0.8)
        
        # Plot predicted values for training period
        ax1.plot(train_dates, results['train_predictions'], 'r--', label='Predicted (Training)', linewidth=2, alpha=0.8)
        
        # Plot predicted values for test period (projections only)
        ax1.plot(test_dates, results['test_predictions'], 'g-', label='Predicted (Projections)', linewidth=2, alpha=0.8)
        
        # Add vertical line to separate training and test periods
        if len(train_dates) > 0 and len(test_dates) > 0:
            split_date = train_dates.max()
            ax1.axvline(x=split_date, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Train/Test Split')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Target Values')
        ax1.set_title('Time Series: Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45)
        
        # Calculate and display performance metrics
        train_r2 = r2_score(results['y_train'], results['train_predictions'])
        train_mse = mean_squared_error(results['y_train'], results['train_predictions'])
        
        # Add text box with performance metrics
        textstr = f'Training R² = {train_r2:.3f}\nTraining MSE = {train_mse:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        print(f"✓ Time-series plot created with {len(train_dates)} training points and {len(test_dates)} projection points")
        
    else:
        ax1.text(0.5, 0.5, 'Date information not available\nfor time-series plot', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Time Series: Data Not Available')
        logger.warning("Date information not available for time-series plot")
    
    # 2. Training Actual vs Predicted Scatter
    ax2 = axes[1]
    ax2.scatter(results['y_train'], results['train_predictions'], alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(results['y_train'].min(), results['train_predictions'].min())
    max_val = max(results['y_train'].max(), results['train_predictions'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Training Set: Actual vs Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display R²
    train_r2 = r2_score(results['y_train'], results['train_predictions'])
    ax2.text(0.05, 0.95, f'R² = {train_r2:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Residuals plot
    ax3 = axes[2]
    residuals = results['y_train'] - results['train_predictions']
    ax3.scatter(results['train_predictions'], residuals, alpha=0.6, s=20)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals Plot')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        # Extract end date from output path or use current date
        if '_' in output_path and output_path.endswith('.csv'):
            # Try to extract date from filename like USA_Baseline_20250101.csv
            filename = output_path.split('\\')[-1]  # Get filename
            if '_' in filename and filename.endswith('.csv'):
                date_part = filename.split('_')[-1].replace('.csv', '')
                plot_path = output_path.replace('.csv', f'_model_results_{date_part}.png')
            else:
                plot_path = output_path.replace('.csv', '_model_results.png')
        else:
            plot_path = output_path.replace('.csv', '_model_results.png')
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {plot_path}")
        print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Training R²: {train_r2:.4f}")
    print(f"Training MSE: {mean_squared_error(results['y_train'], results['train_predictions']):.4f}")
    print(f"Training MAE: {mean_absolute_error(results['y_train'], results['train_predictions']):.4f}")
    print(f"Number of features used: {len(results['features_used'])}")
    print(f"Number of training samples: {len(results['y_train'])}")
    print(f"Number of test samples: {len(results['test_predictions'])}")
    
    # Additional time-series statistics if dates available
    if results['train_dates'] is not None and results['test_dates'] is not None:
        train_period = f"{results['train_dates'].min().strftime('%Y-%m')} to {results['train_dates'].max().strftime('%Y-%m')}"
        test_period = f"{results['test_dates'].min().strftime('%Y-%m')} to {results['test_dates'].max().strftime('%Y-%m')}"
        print(f"Training period: {train_period}")
        print(f"Projection period: {test_period}")
        print(f"Test predictions range: {results['test_predictions'].min():.2f} to {results['test_predictions'].max():.2f}")
    
    logger.info("Plotting complete")
    print("✓ Results plotting complete")

def predictTimeWindow(start_date, end_date, plot_results=True, save_plot=False, output_dir=None):
    """
    Use the trained USA baseline model to predict values for a specific time window and plot fitted vs actual.
    
    Parameters:
    - start_date: Start date for prediction window (YYYY-MM-DD format)
    - end_date: End date for prediction window (YYYY-MM-DD format)
    - plot_results: Whether to create plots (default: True)
    - save_plot: Whether to save the plot (default: False)
    - output_dir: Directory to save plot (if save_plot=True)
    
    Returns:
    - prediction_results: Dictionary with predictions and metrics for the time window
    """
    global USA_RESULTS
    
    if not USA_RESULTS:
        print("❌ No USA baseline results found. Please run the main pipeline first.")
        return None
    
    results = USA_RESULTS
    
    logger.info(f"Predicting time window {start_date} to {end_date} for USA baseline")
    print(f"Predicting time window {start_date} to {end_date} for USA baseline")
    
    try:
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get the original data
        usa_data = results['original_data'].copy()
        date_col = results['date_column']
        
        # Filter data for the specified time window
        mask = (usa_data[date_col] >= start_dt) & (usa_data[date_col] <= end_dt)
        window_data = usa_data[mask].copy()
        
        if len(window_data) == 0:
            print(f"❌ No data found for the specified time window {start_date} to {end_date}")
            return None
        
        print(f"✓ Found {len(window_data)} data points in the specified time window")
        
        # Get the features used in training
        features_used = results['features_used']
        target_col = results['target_column']
        
        # Check if we have the target values for this window (for comparison)
        has_actual_values = target_col in window_data.columns and window_data[target_col].notna().any()
        
        # Prepare features for prediction
        X_window = window_data[features_used].copy()
        
        # Handle any missing values in features (use same approach as training)
        for col in features_used:
            if X_window[col].isnull().any():
                X_window[col] = X_window[col].fillna(X_window[col].median())
        
        # Apply the same transformations used in training
        scaler = results['scaler']
        if scaler is not None:
            X_window_scaled = X_window.copy()
            X_window_scaled[features_used] = scaler.transform(X_window_scaled[features_used])
        else:
            X_window_scaled = X_window.copy()
        
        # Make predictions using the trained meta learner
        meta_learner = results['meta_learner']
        individual_models = results['individual_models']
        
        # Get base model predictions first
        base_predictions = []
        for name, model in individual_models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_window_scaled[features_used])
                else:
                    # For GridSearchCV objects
                    pred = model.best_estimator_.predict(X_window_scaled[features_used])
                base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {str(e)}")
        
        if len(base_predictions) == 0:
            print("❌ Could not get predictions from any base models")
            return None
        
        # Stack base predictions for meta learner
        meta_X = np.column_stack(base_predictions)
        
        # Get final predictions from meta learner
        final_predictions = meta_learner.predict(meta_X)
        
        # Prepare results
        prediction_results = {
            'time_window': (start_date, end_date),
            'dates': window_data[date_col],
            'predictions': final_predictions,
            'features_used': features_used,
            'data_points': len(window_data)
        }
        
        # Add actual values if available
        if has_actual_values:
            actual_values = window_data[target_col].values
            # Filter out NaN values for metrics calculation
            valid_mask = ~np.isnan(actual_values)
            if valid_mask.any():
                valid_actual = actual_values[valid_mask]
                valid_pred = final_predictions[valid_mask]
                
                # Calculate metrics
                mse = mean_squared_error(valid_actual, valid_pred)
                r2 = r2_score(valid_actual, valid_pred)
                mae = mean_absolute_error(valid_actual, valid_pred)
                
                prediction_results['actual_values'] = actual_values
                prediction_results['metrics'] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'valid_points': len(valid_actual)
                }
                
                print(f"✓ Time window metrics - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
            else:
                prediction_results['actual_values'] = actual_values
                prediction_results['metrics'] = None
                print("⚠️  No valid actual values found for metrics calculation")
        else:
            prediction_results['actual_values'] = None
            prediction_results['metrics'] = None
            print("ℹ️  No actual values available for comparison")
        
        # Create plots if requested
        if plot_results:
            plt.style.use('seaborn-v0_8')
            
            if has_actual_values and prediction_results['metrics'] is not None:
                # Create 2x2 subplot if we have actual values
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'USA Baseline - Time Window Prediction Analysis\n{start_date} to {end_date}', 
                           fontsize=14, fontweight='bold')
                
                # 1. Time series plot
                ax1 = axes[0, 0]
                dates = pd.to_datetime(prediction_results['dates'])
                ax1.plot(dates, actual_values, 'b-', label='Actual', linewidth=2, alpha=0.8)
                ax1.plot(dates, final_predictions, 'r--', label='Predicted', linewidth=2, alpha=0.8)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Target Values')
                ax1.set_title('Time Series: Actual vs Predicted')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Add metrics
                metrics = prediction_results['metrics']
                textstr = f'R² = {metrics["r2"]:.3f}\nMSE = {metrics["mse"]:.2f}\nMAE = {metrics["mae"]:.2f}'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
                
                # 2. Scatter plot
                ax2 = axes[0, 1]
                valid_mask = ~np.isnan(actual_values)
                if valid_mask.any():
                    valid_actual = actual_values[valid_mask]
                    valid_pred = final_predictions[valid_mask]
                    ax2.scatter(valid_actual, valid_pred, alpha=0.6, s=30)
                    
                    # Perfect prediction line
                    min_val = min(valid_actual.min(), valid_pred.min())
                    max_val = max(valid_actual.max(), valid_pred.max())
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                ax2.set_xlabel('Actual Values')
                ax2.set_ylabel('Predicted Values')
                ax2.set_title('Scatter: Actual vs Predicted')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. Residuals plot
                ax3 = axes[1, 0]
                if valid_mask.any():
                    residuals = valid_actual - valid_pred
                    ax3.scatter(valid_pred, residuals, alpha=0.6, s=30)
                    ax3.axhline(y=0, color='r', linestyle='--')
                ax3.set_xlabel('Predicted Values')
                ax3.set_ylabel('Residuals')
                ax3.set_title('Residuals Plot')
                ax3.grid(True, alpha=0.3)
                
                # 4. Prediction distribution
                ax4 = axes[1, 1]
                ax4.hist(final_predictions, bins=20, alpha=0.7, color='red', label='Predicted', density=True)
                if valid_mask.any():
                    ax4.hist(valid_actual, bins=20, alpha=0.7, color='blue', label='Actual', density=True)
                ax4.set_xlabel('Values')
                ax4.set_ylabel('Density')
                ax4.set_title('Distribution Comparison')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
            else:
                # Create single plot if no actual values
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                fig.suptitle(f'USA Baseline - Time Window Predictions\n{start_date} to {end_date}', 
                           fontsize=14, fontweight='bold')
                
                dates = pd.to_datetime(prediction_results['dates'])
                ax.plot(dates, final_predictions, 'g-', label='Predicted', linewidth=2, alpha=0.8)
                ax.set_xlabel('Date')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Time Series: Predictions')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add prediction stats
                pred_stats = f'Mean: {final_predictions.mean():.2f}\nStd: {final_predictions.std():.2f}\nMin: {final_predictions.min():.2f}\nMax: {final_predictions.max():.2f}'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, pred_stats, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot and output_dir:
                # Get the end date from USA_RESULTS if available
                usa_end_date = ""
                if USA_RESULTS and 'original_data' in USA_RESULTS:
                    # Try to extract end date from the data or use a default
                    try:
                        date_col = USA_RESULTS.get('date_column', 'Date')
                        if date_col in USA_RESULTS['original_data'].columns:
                            max_date = USA_RESULTS['original_data'][date_col].max()
                            usa_end_date = max_date.strftime('%Y%m%d') if hasattr(max_date, 'strftime') else ""
                    except:
                        usa_end_date = ""
                
                plot_filename = f"usa_timewindow_{start_date}_{end_date}_predictions_{usa_end_date}.png"
                plot_path = f"{output_dir}/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to: {plot_path}")
                print(f"✓ Plot saved to: {plot_path}")
            
            plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"USA BASELINE - TIME WINDOW PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Time Window: {start_date} to {end_date}")
        print(f"Data Points: {len(window_data)}")
        print(f"Features Used: {len(features_used)}")
        print(f"Prediction Range: {final_predictions.min():.2f} to {final_predictions.max():.2f}")
        print(f"Prediction Mean: {final_predictions.mean():.2f}")
        
        if prediction_results['metrics']:
            print(f"\nAccuracy Metrics:")
            metrics = prediction_results['metrics']
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Valid Points: {metrics['valid_points']}")
        
        logger.info(f"Time window prediction complete for USA baseline")
        print("✓ Time window prediction complete")
        
        return prediction_results
        
    except Exception as e:
        logger.error(f"Error in predictTimeWindow: {str(e)}")
        print(f"❌ Error in time window prediction: {str(e)}")
        return None

def createStandardizedOutput(df_filled, train_df, test_df, results, date_col, target_col, hpi_col, new_target_col, new_hpi_col):
    """
    Create standardized output CSV with only the required columns.
    
    Parameters:
    - df_filled: Original filled dataframe
    - train_df: Training dataframe
    - test_df: Test dataframe
    - results: Model results dictionary
    - date_col: Date column name
    - target_col: Original target column name (hpa12m)
    - hpi_col: HPI column name
    - new_target_col: Forward-looking target column name
    - new_hpi_col: Forward-looking HPI column name
    
    Returns:
    - output_df: Standardized output dataframe with required columns
    """
    logger.info("Creating standardized output format...")
    print("Creating standardized output format...")
    
    # Create output dataframe with date column
    output_df = pd.DataFrame()
    output_df['Year_Month_Day'] = df_filled[date_col].dt.strftime('%Y-%m-%d')
    
    # Add predictions column - ProjectedHPA1YFwd_USABaseline (model predictions for 12-month forward hpa12m)
    output_df['ProjectedHPA1YFwd_USABaseline'] = np.nan
    output_df.loc[train_df.index, 'ProjectedHPA1YFwd_USABaseline'] = results['train_predictions']
    output_df.loc[test_df.index, 'ProjectedHPA1YFwd_USABaseline'] = results['test_predictions']
    
    # Add actual values columns
    if target_col and target_col in df_filled.columns:
        # USA_HPA1Yfwd - actual hpa12m values where available (NaN for last 12 months by design)
        output_df['USA_HPA1Yfwd'] = df_filled[target_col]
    else:
        # If no target column specified, create empty column
        output_df['USA_HPA1Yfwd'] = np.nan
    
    # USA_HPI1Yfwd - 12-month forward HPI values where available (NaN for last 12 months by design)
    if hpi_col and hpi_col in df_filled.columns and new_hpi_col in df_filled.columns:
        output_df['USA_HPI1Yfwd'] = df_filled[new_hpi_col]
    else:
        # If no HPI column specified, create empty column
        output_df['USA_HPI1Yfwd'] = np.nan
    
    # Ensure the last 12 months have NaN values for actual columns (by design)
    if len(output_df) >= 12:
        last_12_months_mask = output_df.index >= (len(output_df) - 12)
        output_df.loc[last_12_months_mask, 'USA_HPA1Yfwd'] = np.nan
        output_df.loc[last_12_months_mask, 'USA_HPI1Yfwd'] = np.nan
    
    logger.info(f"Standardized output created with {len(output_df)} rows")
    print(f"✓ Standardized output created with {len(output_df)} rows")
    
    return output_df

# Main execution function
def main():
    """
    Main execution function that runs the entire pipeline.
    """
    global USA_RESULTS
    
    try:
        print("="*60)
        print("GEOGRAPHICAL HOME PRICE RANKER - USA BASELINE")
        print("="*60)
        
        # Get end date from user
        end_date = input("Enter end date (YYYY-MM-DD): ")
        if not end_date:
            raise ValueError("End date is required. Please provide a valid date in YYYY-MM-DD format.")
        
        # Validate date format
        try:
            pd.to_datetime(end_date)
        except:
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-12-31)")
        
        # Step 1: Initialize configuration
        cfg = CFG(get_user_input=False)  # Set to True to get user input
        cfg.end_date = end_date
        
        # Generate output path with end date
        end_date_formatted = end_date.replace('-', '')
        cfg.outputPath = f"D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_{end_date_formatted}.csv"
        
        # Step 2: Load and check data
        df = loadDataAndCheckAllMonths(cfg.filePath, cfg.dateCol, cfg.start_date, cfg.end_date)
        
        # Step 3: Add features
        df_features = addAllFeatures(df, cfg.idList, cfg.dateCol, cfg.featureList, 
                                   cfg.targetCol, cfg.lagList, cfg.movingAverages, cfg.rateList)
        
        # Step 4: Add target
        df_target, new_target_col, new_hpi_col = addTarget(df_features, cfg.idList, cfg.dateCol, 
                                            cfg.targetCol, cfg.hpiCol, cfg.targetForward)
        
        # Step 5: Fill missing values and create train/test split
        df_filled, train_df, test_df, x_columns = fillMissingValues(df_target, new_target_col, 
                                                                  cfg.idList, cfg.dateCol)
        
        # Step 6: Remove skewness and kurtosis
        train_transformed, test_transformed, transformers = removeSkewnessAndKurtosis(
            train_df, test_df, x_columns)
        
        # Step 7: Standardize data
        train_scaled, test_scaled, scaler, numeric_x_columns = standardizeData(train_transformed, test_transformed, x_columns)
        
        # Step 8: Remove high VIF features
        train_clean, test_clean, final_features = checkAndRemoveHighVIF(train_scaled, test_scaled, numeric_x_columns)
        
        # Step 9: Setup CV scheme
        cv_scheme = timeseriesCV()
        
        # Step 10: Train models and create ensemble
        results = fitAndPredictVotingRegressor(train_clean, test_clean, final_features, 
                                             cfg.featuresToUse, new_target_col, 
                                             cfg.AllModelsList, cfg.AllModelParams, cv_scheme, scaler, cfg.dateCol)
        
        # Store results globally for time window predictions
        USA_RESULTS.update(results)
        USA_RESULTS['original_data'] = df_filled
        USA_RESULTS['train_data'] = train_df
        USA_RESULTS['test_data'] = test_df
        USA_RESULTS['scaler'] = scaler
        USA_RESULTS['transformers'] = transformers
        
        # Step 11: Plot results
        plotResults(results, cfg.outputPath)
        
        # Step 12: Create standardized output
        output_df = createStandardizedOutput(df_filled, train_df, test_df, results, 
                                           cfg.dateCol, cfg.targetCol, cfg.hpiCol, new_target_col, new_hpi_col)
        
        # Save standardized results
        output_df.to_csv(cfg.outputPath, index=False)
        logger.info(f"Standardized results saved to: {cfg.outputPath}")
        print(f"✓ Standardized results saved to: {cfg.outputPath}")
        
        # Also save detailed results for debugging (optional)
        detailed_output_path = cfg.outputPath.replace('.csv', '_detailed.csv')
        final_df = df_filled.copy()
        final_df['predictions'] = np.nan
        final_df.loc[train_df.index, 'predictions'] = results['train_predictions']
        final_df.loc[test_df.index, 'predictions'] = results['test_predictions']
        final_df.to_csv(detailed_output_path, index=False)
        logger.info(f"Detailed results saved to: {detailed_output_path}")
        
        print("\n" + "="*50)
        print("USAGE INSTRUCTIONS")
        print("="*50)
        print("To analyze specific time windows, use:")
        print("  predictTimeWindow('START_DATE', 'END_DATE')")
        print("\nExamples:")
        print("  predictTimeWindow('2020-01-01', '2020-12-31')")
        print("  predictTimeWindow('2021-06-01', '2022-05-31')")
        print("  predictTimeWindow('2019-01-01', '2019-12-31', save_plot=True, output_dir='./plots')")
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        raise e

def run_with_custom_paths(usa_file, output_file, end_date, config=None):
    """
    Run the USA baseline model with custom file paths and configurations.
    
    Parameters:
    - usa_file: Path to the input USA data CSV file
    - output_file: Path to save the output results CSV file
    - end_date: End date in YYYY-MM-DD format (required)
    - config: Dictionary containing custom configurations (optional)
              If not provided, default configurations will be used
    
    Returns:
    - None (saves results to output_file)
    """
    global USA_RESULTS
    
    try:
        # Validate end_date parameter
        if not end_date or not isinstance(end_date, str):
            raise ValueError("end_date parameter is required. Please provide a valid date in YYYY-MM-DD format.")
        
        # Validate date format
        try:
            pd.to_datetime(end_date)
        except:
            raise ValueError("Invalid end_date format. Please use YYYY-MM-DD format (e.g., 2024-12-31)")
        
        print("="*60)
        print("GEOGRAPHICAL HOME PRICE RANKER - USA BASELINE (CUSTOM CONFIG)")
        print("="*60)
        
        # Create custom configuration
        class CustomCFG:
            def __init__(self, usa_file, output_file, end_date, config=None):
                # Default configurations
                self.filePath = usa_file
                self.outputPath = output_file
                self.idList = []
                self.dateCol = "Date"
                self.start_date = "1990-01-01"
                self.end_date = end_date
                self.featureList = []
                self.targetCol = "hpa12m"
                self.hpiCol = "HPI"
                self.lagList = [1,3,6,8,12,15,18,24,36,48,60]
                self.rateList = [1,2,3,4,5,6,7,8,9,10,11,12]
                self.movingAverages = [1,3,6,9,12,18,24]
                self.targetForward = 12
                self.featuresToUse = []
                
                # Model configurations
                self.AllModelsList = [
                    'LinearRegression',
                    'Ridge',
                    'Lasso',
                    'ElasticNet',
                    'RandomForest',
                    'XGBoost',
                    'LightGBM',
                    'GradientBoosting'
                ]
                
                self.AllModelParams = {
                    'LinearRegression': {},
                    'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                    'Lasso': {'alpha': [0.1, 1.0, 10.0]},
                    'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
                    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
                    'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
                    'LightGBM': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
                    'GradientBoosting': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
                }
                
                # Override with custom configurations if provided
                if config:
                    for key, value in config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        # Initialize configuration with custom paths and config
        cfg = CustomCFG(usa_file, output_file, end_date, config)
        
        # Step 2: Load and check data
        df = loadDataAndCheckAllMonths(cfg.filePath, cfg.dateCol, cfg.start_date, cfg.end_date)
        
        # Step 3: Add features
        df_features = addAllFeatures(df, cfg.idList, cfg.dateCol, cfg.featureList, 
                                   cfg.targetCol, cfg.lagList, cfg.movingAverages, cfg.rateList)
        
        # Step 4: Add target
        df_target, new_target_col, new_hpi_col = addTarget(df_features, cfg.idList, cfg.dateCol, 
                                            cfg.targetCol, cfg.hpiCol, cfg.targetForward)
        
        # Step 5: Fill missing values and create train/test split
        df_filled, train_df, test_df, x_columns = fillMissingValues(df_target, new_target_col, 
                                                                  cfg.idList, cfg.dateCol)
        
        # Step 6: Remove skewness and kurtosis
        train_transformed, test_transformed, transformers = removeSkewnessAndKurtosis(
            train_df, test_df, x_columns)
        
        # Step 7: Standardize data
        train_scaled, test_scaled, scaler, numeric_x_columns = standardizeData(train_transformed, test_transformed, x_columns)
        
        # Step 8: Remove high VIF features
        train_clean, test_clean, final_features = checkAndRemoveHighVIF(train_scaled, test_scaled, numeric_x_columns)
        
        # Step 9: Setup CV scheme
        cv_scheme = timeseriesCV()
        
        # Step 10: Train models and create ensemble
        results = fitAndPredictVotingRegressor(train_clean, test_clean, final_features, 
                                             cfg.featuresToUse, new_target_col, 
                                             cfg.AllModelsList, cfg.AllModelParams, cv_scheme, scaler, cfg.dateCol)
        
        # Store results globally for time window predictions
        USA_RESULTS.update(results)
        USA_RESULTS['original_data'] = df_filled
        USA_RESULTS['train_data'] = train_df
        USA_RESULTS['test_data'] = test_df
        USA_RESULTS['scaler'] = scaler
        USA_RESULTS['transformers'] = transformers
        
        # Step 11: Plot results
        plotResults(results, cfg.outputPath)
        
        # Step 12: Create standardized output
        output_df = createStandardizedOutput(df_filled, train_df, test_df, results, 
                                           cfg.dateCol, cfg.targetCol, cfg.hpiCol, new_target_col, new_hpi_col)
        
        # Save standardized results
        output_df.to_csv(cfg.outputPath, index=False)
        logger.info(f"Standardized results saved to: {cfg.outputPath}")
        print(f"✓ Standardized results saved to: {cfg.outputPath}")
        
        # Also save detailed results for debugging (optional)
        detailed_output_path = cfg.outputPath.replace('.csv', '_detailed.csv')
        final_df = df_filled.copy()
        final_df['predictions'] = np.nan
        final_df.loc[train_df.index, 'predictions'] = results['train_predictions']
        final_df.loc[test_df.index, 'predictions'] = results['test_predictions']
        final_df.to_csv(detailed_output_path, index=False)
        logger.info(f"Detailed results saved to: {detailed_output_path}")
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE!")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        return False

"""
# ==============================================================================
# EXAMPLE USAGE WITH CUSTOM PATHS AND CONFIGURATIONS
# ==============================================================================
# 
# Example 1: Basic usage with custom paths
# run_with_custom_paths(
#     usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
#     output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_20250101.csv",
#     end_date="2025-01-01"
# )
# 
# Example 2: Advanced usage with custom configurations
# custom_config = {
#     'start_date': '2010-01-01',
#     'end_date': '2024-12-31',
#     'featureList': ['unemployment_rate', 'interest_rate', 'gdp_growth', 'inflation_rate'],
#     'targetCol': 'hpa12m',
#     'hpiCol': 'HPI',
#     'lagList': [1, 3, 6, 12, 24],
#     'rateList': [1, 3, 6, 12],
#     'movingAverages': [3, 6, 12],
#     'targetForward': 12,
#     'featuresToUse': ['unemployment_rate', 'interest_rate'],
#     'AllModelsList': ['LinearRegression', 'Ridge', 'RandomForest', 'XGBoost'],
#     'AllModelParams': {
#         'LinearRegression': {},
#         'Ridge': {'alpha': [0.1, 1.0, 10.0]},
#         'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
#         'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
#     }
# }
# 
# run_with_custom_paths(
#     usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
#     output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_Custom_20250101.csv",
#     end_date="2024-12-31",
#     config=custom_config
# )
# 
# Example 3: Minimal configuration with only essential parameters
# minimal_config = {
#     'start_date': '2020-01-01',
#     'end_date': '2024-12-31',
#     'featureList': ['unemployment_rate', 'interest_rate'],
#     'targetCol': 'hpa12m',
#     'hpiCol': 'HPI',
#     'AllModelsList': ['LinearRegression', 'RandomForest']
# }
# 
# run_with_custom_paths(
#     usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
#     output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_Minimal_20250101.csv",
#     end_date="2024-12-31",
#     config=minimal_config
# )
"""

if __name__ == "__main__":
    main()


'''
run_with_custom_paths(
    usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
    output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_20250101.csv",
    end_date="2025-01-01"
)
custom_config = {
    'start_date': '2010-01-01',
    'end_date': '2024-12-31',
    'featureList': ['unemployment_rate', 'interest_rate', 'gdp_growth', 'inflation_rate'],
    'targetCol': 'hpa12m',
    'hpiCol': 'HPI',
    'lagList': [1, 3, 6, 12, 24],
    'rateList': [1, 3, 6, 12],
    'movingAverages': [3, 6, 12],
    'targetForward': 12,
    'featuresToUse': ['unemployment_rate', 'interest_rate'],
    'AllModelsList': ['LinearRegression', 'Ridge', 'RandomForest', 'XGBoost'],
    'AllModelParams': {
        'LinearRegression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
    }
}
run_with_custom_paths(
    usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
    output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_Custom_20250101.csv",
    end_date="2024-12-31",
    config=custom_config
)
minimal_config = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'featureList': ['unemployment_rate', 'interest_rate'],
    'targetCol': 'hpa12m',
    'hpiCol': 'HPI',
    'AllModelsList': ['LinearRegression', 'RandomForest']
}

run_with_custom_paths(
    usa_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\unified_monthly_data.csv",
    output_file="D:\\Geographical-Home-Price-Ranker\\AllProcessedFIles\\USA_Baseline_Minimal_20250101.csv",
    end_date="2024-12-31",
    config=minimal_config
)
'''