import pandas as pd
import numpy as np
import warnings
import logging
import os
from pathlib import Path
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

class CFG:
    """
    Configuration class for MSA Calibration model.
    """
    
    def __init__(self, get_user_input=True):
        logger.info("Initializing CFG class for MSA Calibration...")
        
        if get_user_input:
            self._get_user_configurations()
        else:
            self._set_default_configurations()
        
        self._setup_plotting_specs()
        logger.info("CFG initialization complete.")
    
    def _get_user_configurations(self):
        """Get all configurations from user input"""
        print("=== MSA CALIBRATION CONFIGURATION SETUP ===")
        
        # Data paths and columns
        self.msa_baseline_path = input("Enter MSA baseline data CSV file path: ")
        self.msa_new_data_path = input("Enter new MSA data CSV file path: ")
        self.output_path = input("Enter output CSV file path [MSA_Calibration_Results.csv]: ") or "MSA_Calibration_Results.csv"
        
        # Column configurations
        self.date_col = input("Enter date column name: ")
        self.id_columns = input("Enter ID columns to merge on (comma-separated): ").split(",")
        
        # MSA Baseline columns
        print("\nMSA Baseline Columns:")
        self.msa_baseline_columns = input("Enter MSA baseline columns to use (comma-separated): ").split(",")
        self.target_column = input("Enter target column name from MSA baseline: ")
        self.msa_hpi_col = input("Enter MSA HPI column name from MSA baseline: ")
        self.msa_hpa12m_col = input("Enter MSA HPA12M column name from MSA baseline: ")
        self.usa_hpi12mF_col = input("Enter USA HPI forward column name from MSA baseline: ")
        self.usa_hpa12mF_col = input("Enter USA HPA forward column name from MSA baseline: ")
        self.hpi1y_fwd_col = input("Enter HPI1Y forward column name from MSA baseline: ")
        self.usa_projection_col = input("Enter USA projection column name from MSA baseline: ")
        self.msa_projection_col = input("Enter MSA projection column name from MSA baseline: ")
        
        # MSA New Data columns
        print("\nMSA New Data Columns:")
        self.msa_new_columns = input("Enter MSA new data columns to use (comma-separated): ").split(",")
        
        # Date configurations
        self.start_date = input("Enter start date (YYYY-MM-DD) [1990-01-01]: ") or "1990-01-01"
        self.end_date = input("Enter end date (YYYY-MM-DD) [2025-01-01]: ") or "2025-01-01"
        
        # Model configurations
        self._setup_model_configurations()
    
    def _set_default_configurations(self):
        """Set default configurations"""
        self.msa_baseline_path = "MSA_Baseline_Results.csv"
        self.msa_new_data_path = "msa_data.csv"
        self.output_path = "MSA_Calibration_Results.csv"
        
        # Column names
        self.date_col = "Year_Month_Day"
        self.id_columns = ["rcode", "cs_name"]
        self.msa_baseline_columns = []  # No default baseline columns
        self.target_column = "HPA1Yfwd"
        self.msa_hpi_col = "HPI"
        self.msa_hpa12m_col = "hpa12m"
        self.usa_hpi12mF_col = "USA_HPI1Yfwd"
        self.usa_hpa12mF_col = "USA_HPA1Yfwd"
        self.hpi1y_fwd_col = "HPI1Y_fwd"
        self.usa_projection_col = "ProjectedHPA1YFwd_USABaseline"
        self.msa_projection_col = "ProjectedHPA1YFwd_MSABaseline"
        self.msa_new_columns = []
        
        # Date range
        self.start_date = "1990-01-01"
        self.end_date = "2025-01-01"
        
        # Model configurations
        self._setup_model_configurations()
    
    def _setup_model_configurations(self):
        """Setup model configurations"""
        self.AllModelsList = [
            'Ridge',
            'RandomForest',
            'XGBoost'
        ]
        
        self.AllModelParams = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0]},
            'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
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

# Global variable to store region results
REGION_RESULTS = {}

def validateAndPreparePaths(cfg):
    """
    Validate input file paths and prepare output directory.
    
    Parameters:
    - cfg: Configuration object
    
    Returns:
    - bool: True if all paths are valid, raises exception otherwise
    """
    logger.info("Validating file paths and preparing directories...")
    print("Validating file paths and preparing directories...")
    
    try:
        # Convert paths to Path objects for better handling
        msa_path = Path(cfg.msa_baseline_path)
        usa_path = Path(cfg.msa_new_data_path)
        output_path = Path(cfg.output_path)
        
        # Validate input files exist
        if not msa_path.exists():
            error_msg = f"❌ MSA baseline data file not found: {cfg.msa_baseline_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not usa_path.exists():
            error_msg = f"❌ New MSA data file not found: {cfg.msa_new_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate input files are readable
        if not os.access(msa_path, os.R_OK):
            error_msg = f"❌ MSA baseline data file is not readable: {cfg.msa_baseline_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        if not os.access(usa_path, os.R_OK):
            error_msg = f"❌ New MSA data file is not readable: {cfg.msa_new_data_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        # Validate file extensions
        valid_extensions = ['.csv', '.CSV']
        if msa_path.suffix not in valid_extensions:
            error_msg = f"❌ MSA baseline data file must be CSV format: {cfg.msa_baseline_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if usa_path.suffix not in valid_extensions:
            error_msg = f"❌ New MSA data file must be CSV format: {cfg.msa_new_data_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check file sizes (warn if very large)
        msa_size_mb = msa_path.stat().st_size / (1024 * 1024)
        usa_size_mb = usa_path.stat().st_size / (1024 * 1024)
        
        if msa_size_mb > 500:  # 500 MB
            logger.warning(f"⚠️  Large MSA baseline data file detected: {msa_size_mb:.1f} MB. Processing may take longer.")
            print(f"⚠️  Large MSA baseline data file detected: {msa_size_mb:.1f} MB. Processing may take longer.")
        
        if usa_size_mb > 100:  # 100 MB
            logger.warning(f"⚠️  Large new MSA data file detected: {usa_size_mb:.1f} MB. Processing may take longer.")
            print(f"⚠️  Large new MSA data file detected: {usa_size_mb:.1f} MB. Processing may take longer.")
        
        # Prepare output directory
        output_dir = output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
                print(f"✓ Created output directory: {output_dir}")
            except PermissionError:
                error_msg = f"❌ Cannot create output directory: {output_dir}. Permission denied."
                logger.error(error_msg)
                raise PermissionError(error_msg)
        
        # Check if output directory is writable
        if not os.access(output_dir, os.W_OK):
            error_msg = f"❌ Output directory is not writable: {output_dir}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        # Check if output file already exists (warn but allow overwrite)
        if output_path.exists():
            logger.warning(f"⚠️  Output file already exists and will be overwritten: {cfg.output_path}")
            print(f"⚠️  Output file already exists and will be overwritten: {cfg.output_path}")
            
            # Check if existing file is writable
            if not os.access(output_path, os.W_OK):
                error_msg = f"❌ Cannot overwrite existing output file: {cfg.output_path}. Permission denied."
                logger.error(error_msg)
                raise PermissionError(error_msg)
        
        # Validate output file extension
        if output_path.suffix.lower() not in ['.csv']:
            logger.warning(f"⚠️  Output file extension should be .csv: {cfg.output_path}")
            print(f"⚠️  Output file extension should be .csv: {cfg.output_path}")
        
        # Update config with normalized paths (absolute paths)
        cfg.msa_baseline_path = str(msa_path.resolve())
        cfg.msa_new_data_path = str(usa_path.resolve())
        cfg.output_path = str(output_path.resolve())
        
        logger.info("✓ All file paths validated successfully")
        print("✓ All file paths validated successfully")
        print(f"  MSA baseline data: {cfg.msa_baseline_path} ({msa_size_mb:.1f} MB)")
        print(f"  New MSA data: {cfg.msa_new_data_path} ({usa_size_mb:.1f} MB)")
        print(f"  Output: {cfg.output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Path validation failed: {str(e)}")
        print(f"❌ Path validation failed: {str(e)}")
        raise e

def validateDataIntegrity(msa_df, usa_df, cfg):
    """
    Validate the integrity and format of loaded data.
    
    Parameters:
    - msa_df: MSA dataframe
    - usa_df: USA dataframe  
    - cfg: Configuration object
    
    Returns:
    - bool: True if data is valid, raises exception otherwise
    """
    logger.info("Validating data integrity and format...")
    print("Validating data integrity and format...")
    
    try:
        # Check if dataframes are not empty
        if msa_df.empty:
            error_msg = "❌ MSA baseline data file is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if usa_df.empty:
            error_msg = "❌ New MSA data file is empty"  
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check required columns exist in MSA data
        required_msa_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col, cfg.hpi_col, cfg.hpa12m_col]
        missing_msa_cols = [col for col in required_msa_cols if col not in msa_df.columns]
        
        if missing_msa_cols:
            error_msg = f"❌ Missing required columns in MSA baseline data: {missing_msa_cols}"
            logger.error(error_msg)
            print(f"Available MSA baseline data columns: {list(msa_df.columns)}")
            raise ValueError(error_msg)
        
        # Check required columns exist in USA data
        required_usa_cols = [cfg.date_col]
        missing_usa_cols = [col for col in required_usa_cols if col not in usa_df.columns]
        
        if missing_usa_cols:
            error_msg = f"❌ Missing required columns in new MSA data: {missing_usa_cols}"
            logger.error(error_msg)
            print(f"Available new MSA data columns: {list(usa_df.columns)}")
            raise ValueError(error_msg)
        
        # Check date column format
        try:
            pd.to_datetime(msa_df[cfg.date_col])
        except Exception:
            error_msg = f"❌ Invalid date format in MSA baseline data column '{cfg.date_col}'"
            logger.error(error_msg)
            print(f"Sample MSA baseline data date values: {msa_df[cfg.date_col].head().tolist()}")
            raise ValueError(error_msg)
        
        try:
            pd.to_datetime(usa_df[cfg.date_col])
        except Exception:
            error_msg = f"❌ Invalid date format in new MSA data column '{cfg.date_col}'"
            logger.error(error_msg)
            print(f"Sample new MSA data date values: {usa_df[cfg.date_col].head().tolist()}")
            raise ValueError(error_msg)
        
        # Check for minimum data requirements
        if len(msa_df) < 12:
            logger.warning(f"⚠️  MSA baseline data has very few records ({len(msa_df)}). Results may be unreliable.")
            print(f"⚠️  MSA baseline data has very few records ({len(msa_df)}). Results may be unreliable.")
        
        if len(usa_df) < 12:
            logger.warning(f"⚠️  new MSA data has very few records ({len(usa_df)}). Results may be unreliable.")
            print(f"⚠️  new MSA data has very few records ({len(usa_df)}). Results may be unreliable.")
        
        # Check for duplicate region codes in MSA data
        unique_regions = msa_df[cfg.rcode_col].nunique()
        total_msa_records = len(msa_df)
        
        if unique_regions == 0:
            error_msg = "❌ No valid MSA regions found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✓ Data integrity validation passed")
        print("✓ Data integrity validation passed")
        print(f"  MSA regions: {unique_regions}")
        print(f"  MSA baseline data records: {total_msa_records}")
        print(f"  new MSA data records: {len(usa_df)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data integrity validation failed: {str(e)}")
        print(f"❌ Data integrity validation failed: {str(e)}")
        raise e

def loadAndMergeData(cfg):
    """
    Load and merge MSA baseline data with new MSA data.
    
    Parameters:
    - cfg: Configuration object
    
    Returns:
    - merged_df: Merged dataframe
    - unique_regions: List of unique MSA regions
    """
    logger.info("Loading and merging data...")
    
    try:
        # Load MSA baseline data
        msa_baseline_df = pd.read_csv(cfg.msa_baseline_path)
        logger.info(f"MSA baseline data loaded. Shape: {msa_baseline_df.shape}")
        
        # Load new MSA data
        msa_new_df = pd.read_csv(cfg.msa_new_data_path)
        logger.info(f"New MSA data loaded. Shape: {msa_new_df.shape}")
        
        # Convert date columns to datetime
        msa_baseline_df[cfg.date_col] = pd.to_datetime(msa_baseline_df[cfg.date_col])
        msa_new_df[cfg.date_col] = pd.to_datetime(msa_new_df[cfg.date_col])
        
        # Filter data by date range
        start_dt = pd.to_datetime(cfg.start_date)
        end_dt = pd.to_datetime(cfg.end_date)
        
        msa_baseline_df = msa_baseline_df[(msa_baseline_df[cfg.date_col] >= start_dt) & 
                                        (msa_baseline_df[cfg.date_col] <= end_dt)].copy()
        msa_new_df = msa_new_df[(msa_new_df[cfg.date_col] >= start_dt) & 
                               (msa_new_df[cfg.date_col] <= end_dt)].copy()
        
        # Calculate current USA HPI and HPA by shifting forward-looking columns
        msa_baseline_df[cfg.usa_hpi_col] = msa_baseline_df.groupby(cfg.id_columns[0])[cfg.usa_hpi12mF_col].shift(-12)
        msa_baseline_df[cfg.usa_hpa12m_col] = msa_baseline_df.groupby(cfg.id_columns[0])[cfg.usa_hpa12mF_col].shift(-12)
        
        # Select required columns from MSA baseline
        required_baseline_cols = cfg.id_columns + [cfg.date_col] + cfg.msa_baseline_columns + [
            cfg.target_column,
            cfg.usa_hpi_col,
            cfg.usa_hpa12m_col,
            cfg.usa_hpi12mF_col,
            cfg.usa_hpa12mF_col
        ]
        
        # Verify required columns exist in MSA baseline
        missing_cols = [col for col in required_baseline_cols if col not in msa_baseline_df.columns]
        if missing_cols:
            error_msg = f"❌ Missing required columns in MSA baseline data: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Select only the required columns from MSA baseline
        msa_baseline_df = msa_baseline_df[required_baseline_cols]
        
        # Select required columns from new MSA data
        required_new_cols = cfg.id_columns + [cfg.date_col] + cfg.msa_new_columns
        
        # Verify required columns exist in new MSA data
        missing_cols = [col for col in required_new_cols if col not in msa_new_df.columns]
        if missing_cols:
            error_msg = f"❌ Missing required columns in new MSA data: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Select only the required columns from new MSA data
        msa_new_df = msa_new_df[required_new_cols]
        
        # Drop regions with missing projections in MSA baseline
        regions_with_excessive_missing = []
        for region in msa_baseline_df[cfg.id_columns[0]].unique():
            region_data = msa_baseline_df[msa_baseline_df[cfg.id_columns[0]] == region]
            missing_count = region_data[cfg.target_column].isnull().sum()
            if missing_count > 12:  # Only remove if more than 12 months are missing
                regions_with_excessive_missing.append(region)
        
        if regions_with_excessive_missing:
            logger.warning(f"Found {len(regions_with_excessive_missing)} regions with excessive missing data (>12 months)")
            print(f"\n⚠️  Found {len(regions_with_excessive_missing)} regions with excessive missing data (>12 months):")
            for region in regions_with_excessive_missing:
                print(f"  - Region: {region}")
            
            # Remove regions with excessive missing data
            msa_baseline_df = msa_baseline_df[~msa_baseline_df[cfg.id_columns[0]].isin(regions_with_excessive_missing)]
            logger.info(f"Removed {len(regions_with_excessive_missing)} regions with excessive missing data")
            print(f"✓ Removed {len(regions_with_excessive_missing)} regions with excessive missing data")
        
        # Merge data on ID columns and date
        merge_cols = cfg.id_columns + [cfg.date_col]
        merged_df = pd.merge(
            msa_baseline_df,
            msa_new_df,
            on=merge_cols,
            how='inner'
        )
        
        # Get unique regions
        unique_regions = merged_df[cfg.id_columns[0]].unique()
        
        logger.info(f"Data merged successfully. Shape: {merged_df.shape}")
        logger.info(f"Found {len(unique_regions)} unique MSA regions")
        print(f"\n✓ Data merged successfully")
        print(f"✓ Final dataset contains {len(unique_regions)} unique MSA regions")
        print(f"✓ Final dataset shape: {merged_df.shape}")
        
        return merged_df, unique_regions
        
    except Exception as e:
        logger.error(f"Error in loadAndMergeData: {str(e)}")
        raise e

def createForwardLookingVariables(df, cfg):
    """
    Create forward-looking target variables for each region.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Dataframe with forward-looking variables added
    """
    logger.info("Creating forward-looking target variables...")
    print("Creating forward-looking target variables...")
    
    df = df.copy()
    
    # Sort by region and date
    df = df.sort_values([cfg.rcode_col, cfg.date_col])
    
    # Create 12-month forward HPA12M target
    df['HPA1Yfwd'] = df.groupby(cfg.rcode_col)[cfg.hpa12m_col].shift(-12)
    
    # Create 12-month forward HPI target
    df['HPI1Y_fwd'] = df.groupby(cfg.rcode_col)[cfg.hpi_col].shift(-12)
    
    # Remove rows where we don't have future values (last 12 months)
    df = df.dropna(subset=['HPA1Yfwd', 'HPI1Y_fwd'])
    
    logger.info(f"Forward-looking variables created. Shape: {df.shape}")
    print(f"✓ Forward-looking variables created")
    print(f"✓ Final shape after creating targets: {df.shape}")
    
    return df

def createTrainTestTags(df, cfg):
    """
    Create train/test tags for the data.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Dataframe with train/test tags
    """
    logger.info("Creating train/test tags...")
    
    df = df.copy()
    
    # Sort by region and date
    df = df.sort_values([cfg.rcode_col, cfg.date_col])
    
    # Get the last 12 months for each region
    df['tag'] = 'train'
    for region in df[cfg.rcode_col].unique():
        region_dates = df[df[cfg.rcode_col] == region][cfg.date_col].sort_values()
        if len(region_dates) > 12:
            test_dates = region_dates.iloc[-12:]
            df.loc[df[cfg.date_col].isin(test_dates), 'tag'] = 'test'
    
    return df

def addAllFeatures(df, cfg):
    """
    Add all lagged features, moving averages, and other engineered features.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Enhanced dataframe with new features
    """
    logger.info("Adding engineered features...")
    
    df_enhanced = df.copy()
    
    # Define features to engineer
    base_features = [
        cfg.hpa12m_col + '_baseline',
        cfg.hpi_col + '_baseline',
        'ProjectedHPA1YFwd_USABaseline',
        'ProjectedHPA1YFwd_MSABaseline',
        cfg.hpa12m_col + '_new',
        cfg.hpi_col + '_new'
    ]
    
    # Add any additional features from config
    base_features.extend([f + '_new' for f in cfg.additional_features])
    
    # Remove duplicates and ensure columns exist
    base_features = list(set([f for f in base_features if f in df_enhanced.columns]))
    
    # Sort by region and date for proper lag calculation
    df_enhanced = df_enhanced.sort_values([cfg.rcode_col, cfg.date_col])
    
    # Add lagged features
    for feature in base_features:
        for lag in [1, 3, 6, 12]:
            df_enhanced[f'{feature}_lag_{lag}'] = df_enhanced.groupby(cfg.rcode_col)[feature].shift(lag)
    
    # Add moving averages
    for feature in base_features:
        for window in [3, 6, 12]:
            df_enhanced[f'{feature}_ma_{window}'] = df_enhanced.groupby(cfg.rcode_col)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    return df_enhanced

def processRegionMSA(region, df_region, cfg, target_col='HPA1Yfwd'):
    """
    Process a single MSA region through the modeling pipeline.
    
    Parameters:
    - region: MSA region identifier
    - df_region: Data for specific MSA region
    - cfg: Configuration object
    - target_col: Target column name
    
    Returns:
    - region_results: Dictionary containing all results for the region
    """
    logger.info(f"Processing MSA region: {region}")
    
    try:
        # Check if region has sufficient data
        if df_region.shape[0] < 24:  # At least 2 years of data
            logger.warning(f"Region {region} has insufficient data ({df_region.shape[0]} rows), skipping...")
            return None
        
        # Separate train and test data
        train_df = df_region[df_region['tag'] == 'train'].copy()
        test_df = df_region[df_region['tag'] == 'test'].copy()
        
        # Check if we have enough training data
        if len(train_df) < 12:  # At least 1 year of training data
            logger.warning(f"Region {region} has insufficient training data ({len(train_df)} rows), skipping...")
            return None
        
        # Get feature columns (exclude ID, date, target, and tag columns)
        exclude_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col, 'tag', target_col, 
                       'HPA1Yfwd', 'HPI1Y_fwd', cfg.hpa12m_col + '_baseline', cfg.hpi_col + '_baseline',
                       cfg.hpa12m_col + '_new', cfg.hpi_col + '_new']
        x_columns = [col for col in train_df.columns if col not in exclude_cols and not train_df[col].isnull().all()]
        
        # Check if we have target values in training data
        if target_col not in train_df.columns or train_df[target_col].isnull().all():
            logger.warning(f"Region {region} has no valid target values, skipping...")
            return None
        
        # Remove rows with NaN target values from training data
        train_df = train_df.dropna(subset=[target_col])
        
        if len(train_df) < 12:
            logger.warning(f"Region {region} has insufficient training data after removing NaN targets, skipping...")
            return None
        
        # Fill missing values in features
        for col in x_columns:
            if train_df[col].isnull().any():
                train_df[col] = train_df[col].fillna(train_df[col].median())
                test_df[col] = test_df[col].fillna(train_df[col].median())
        
        # Prepare X and y
        X_train = train_df[x_columns]
        y_train = train_df[target_col]
        X_test = test_df[x_columns]
        
        # Check for sufficient feature diversity
        if len(x_columns) < 3:
            logger.warning(f"Region {region} has too few features ({len(x_columns)}), skipping...")
            return None
        
        # Standardize features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Simple model selection
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        # Train models and get predictions
        model_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                model_predictions[name] = {
                    'train': train_pred,
                    'test': test_pred
                }
                
                # Calculate training score
                r2 = r2_score(y_train, train_pred)
                model_scores[name] = r2
                
                logger.info(f"{name} - Training R²: {r2:.4f}")
                
            except Exception as e:
                logger.warning(f"Error training {name} for region {region}: {str(e)}")
        
        if not model_predictions:
            logger.warning(f"No models successfully trained for region {region}")
            return None
        
        # Create ensemble prediction (simple average)
        if len(model_predictions) > 1:
            ensemble_train = np.mean([pred['train'] for pred in model_predictions.values()], axis=0)
            ensemble_test = np.mean([pred['test'] for pred in model_predictions.values()], axis=0)
        else:
            # Use the single model prediction
            single_model = list(model_predictions.values())[0]
            ensemble_train = single_model['train']
            ensemble_test = single_model['test']
        
        # Calculate performance metrics
        train_r2 = r2_score(y_train, ensemble_train)
        train_mse = mean_squared_error(y_train, ensemble_train)
        train_mae = mean_absolute_error(y_train, ensemble_train)
        
        # Compile results
        results = {
            'region': region,
            'data_shape': df_region.shape,
            'train_predictions': ensemble_train,
            'test_predictions': ensemble_test,
            'model_scores': model_scores,
            'model_predictions': model_predictions,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'features_used': x_columns,
            'target_column': target_col,
            'train_dates': train_df[cfg.date_col],
            'test_dates': test_df[cfg.date_col],
            'scaler': scaler,
            'train_data': train_df,
            'test_data': test_df,
            'performance_metrics': {
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing region {region}: {str(e)}")
        return None

def generateFinalOutput(merged_df, cfg):
    """
    Generate the final output dataframe with all required columns.
    
    Parameters:
    - merged_df: The merged dataframe with all data
    - cfg: Configuration object
    
    Returns:
    - final_df: Final output dataframe with required columns
    """
    logger.info("Generating final output...")
    
    # Start with the base dataframe
    final_df = merged_df.copy()
    
    # Initialize prediction columns
    final_df['ProjectedHPA1YFwd_MSA'] = np.nan
    final_df['Approach3_MSA_HPA1YrFwd'] = np.nan
    
    # Fill in MSA projections from region results
    for region, results in REGION_RESULTS.items():
        if results is None:
            continue
        
        try:
            # Get region mask
            region_mask = final_df[cfg.id_columns[0]] == region
            
            # Fill training predictions
            train_indices = results['train_data'].index
            for idx, pred in zip(train_indices, results['train_predictions']):
                if idx in final_df.index:
                    final_df.loc[idx, 'ProjectedHPA1YFwd_MSA'] = pred
                    final_df.loc[idx, 'Approach3_MSA_HPA1YrFwd'] = pred
            
            # Fill test predictions
            test_indices = results['test_data'].index
            for idx, pred in zip(test_indices, results['test_predictions']):
                if idx in final_df.index:
                    final_df.loc[idx, 'ProjectedHPA1YFwd_MSA'] = pred
                    final_df.loc[idx, 'Approach3_MSA_HPA1YrFwd'] = pred
                    
        except Exception as e:
            logger.warning(f"Error filling predictions for region {region}: {str(e)}")
    
    # Set NaN values for the last 12 months for each region
    for region in final_df[cfg.id_columns[0]].unique():
        region_mask = final_df[cfg.id_columns[0]] == region
        region_dates = final_df[region_mask][cfg.date_col].sort_values()
        if len(region_dates) > 12:
            last_12_months = region_dates.iloc[-12:]
            mask = (final_df[cfg.id_columns[0]] == region) & (final_df[cfg.date_col].isin(last_12_months))
            final_df.loc[mask, [cfg.target_column, cfg.hpi1y_fwd_col, cfg.usa_hpi12mF_col, cfg.usa_hpa12mF_col]] = np.nan
    
    # Select and order the required columns
    required_columns = [
        cfg.date_col,
        cfg.id_columns[0],  # rcode
        cfg.id_columns[1],  # cs_name
        'tag',
        cfg.usa_projection_col,  # USA projection
        cfg.msa_projection_col,  # MSA projection
        cfg.msa_hpi_col,
        cfg.msa_hpa12m_col,
        cfg.target_column,
        cfg.hpi1y_fwd_col,
        cfg.usa_hpi12mF_col,
        cfg.usa_hpa12mF_col,
        'ProjectedHPA1YFwd_MSA',
        'Approach3_MSA_HPA1YrFwd'
    ]
    
    # Select only the required columns
    final_df = final_df[required_columns]
    
    # Convert date to YYYY-MM-01 format
    final_df[cfg.date_col] = pd.to_datetime(final_df[cfg.date_col]).dt.to_period('M').dt.start_time
    
    # Sort by region and date
    final_df = final_df.sort_values([cfg.id_columns[0], cfg.date_col])
    
    return final_df

def fillMissingDataByRegion(df, cfg):
    """
    Fill missing data rows using growth/decay rates region-wise.
    If a region has no data for entire time range, use column averages for each month.
    
    Parameters:
    - df: Input dataframe with potential missing data
    - cfg: Configuration object
    
    Returns:
    - df_filled: Dataframe with missing data filled
    """
    logger.info("Filling missing data by region...")
    print("Filling missing data using growth/decay rates and monthly averages...")
    
    df_filled = df.copy()
    
    # Define numeric columns to fill (exclude ID and date columns)
    exclude_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col]
    numeric_cols = [col for col in df_filled.select_dtypes(include=['float64', 'int64']).columns 
                   if col not in exclude_cols]
    
    print(f"Processing {len(numeric_cols)} numeric columns for missing data...")
    
    # Sort data by region and date
    df_filled = df_filled.sort_values([cfg.rcode_col, cfg.date_col])
    
    # Get all unique regions
    unique_regions = df_filled[cfg.rcode_col].unique()
    unique_regions = [r for r in unique_regions if pd.notna(r)]
    
    # Calculate monthly averages across all regions for fallback
    print("Calculating monthly averages for fallback imputation...")
    df_filled['month'] = pd.to_datetime(df_filled[cfg.date_col]).dt.month
    monthly_averages = {}
    
    for col in numeric_cols:
        monthly_avg = df_filled.groupby('month')[col].mean()
        monthly_averages[col] = monthly_avg
    
    regions_processed = 0
    regions_with_missing = 0
    
    # Process each region
    for region in unique_regions:
        region_mask = df_filled[cfg.rcode_col] == region
        region_df = df_filled[region_mask].copy()
        
        region_has_missing = False
        
        # Check each numeric column for missing data
        for col in numeric_cols:
            col_data = region_df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            missing_count = col_data.isnull().sum()
            
            if missing_count > 0:
                region_has_missing = True
            
            # Check if entire column is missing for this region
            if missing_count == len(region_df):
                # Fill with monthly averages
                print(f"  Region {region}, column {col}: Filling {missing_count} missing values with monthly averages")
                for idx, row in region_df.iterrows():
                    month = row['month']
                    if month in monthly_averages[col] and pd.notna(monthly_averages[col][month]):
                        df_filled.loc[idx, col] = monthly_averages[col][month]
                    else:
                        # If monthly average is also NaN, use overall column mean
                        overall_mean = df_filled[col].mean()
                        if pd.notna(overall_mean):
                            df_filled.loc[idx, col] = overall_mean
                        else:
                            df_filled.loc[idx, col] = 0  # Last resort
            else:
                # Fill using interpolation and growth rates
                region_series = col_data.copy()
                # First, try linear interpolation for gaps within the series
                region_series_interp = region_series.interpolate(method='linear')
                # For remaining NaNs at the beginning or end, use forward/backward fill
                if region_series_interp.isnull().any():
                    # Forward fill for leading NaNs
                    region_series_interp = region_series_interp.fillna(method='ffill')
                    # Backward fill for trailing NaNs
                    region_series_interp = region_series_interp.fillna(method='bfill')
                # If still NaNs (shouldn't happen but safety check), use monthly averages
                if region_series_interp.isnull().any():
                    for idx in region_series_interp[region_series_interp.isnull()].index:
                        month = df_filled.loc[idx, 'month']
                        if month in monthly_averages[col] and pd.notna(monthly_averages[col][month]):
                            region_series_interp.loc[idx] = monthly_averages[col][month]
                        else:
                            overall_mean = df_filled[col].mean()
                            region_series_interp.loc[idx] = overall_mean if pd.notna(overall_mean) else 0
                # Update the main dataframe
                df_filled.loc[region_mask, col] = region_series_interp.values
                print(f"  Region {region}, column {col}: Filled {missing_count} missing values using interpolation")
        
        if region_has_missing:
            regions_with_missing += 1
        
        regions_processed += 1
        
        if regions_processed % 10 == 0:
            print(f"  Processed {regions_processed}/{len(unique_regions)} regions...")
    
    # Remove the temporary month column
    df_filled = df_filled.drop('month', axis=1)
    
    # Final check for any remaining missing values
    remaining_missing = df_filled[numeric_cols].isnull().sum().sum()
    
    if remaining_missing > 0:
        print(f"⚠️  Warning: {remaining_missing} missing values still remain. Applying final cleanup...")
        
        # Final cleanup: fill any remaining NaNs with column means
        for col in numeric_cols:
            col_missing = df_filled[col].isnull().sum()
            if col_missing > 0:
                col_mean = df_filled[col].mean()
                if pd.notna(col_mean):
                    df_filled[col] = df_filled[col].fillna(col_mean)
                else:
                    df_filled[col] = df_filled[col].fillna(0)
                print(f"  Final cleanup: Filled {col_missing} remaining NaNs in {col}")
    
    # Verify no missing values remain in numeric columns
    final_missing = df_filled[numeric_cols].isnull().sum().sum()
    
    logger.info(f"Missing data imputation complete. Regions processed: {regions_processed}")
    logger.info(f"Regions with missing data: {regions_with_missing}")
    logger.info(f"Final missing values: {final_missing}")
    
    print(f"✓ Missing data imputation complete")
    print(f"✓ Processed {regions_processed} regions")
    print(f"✓ {regions_with_missing} regions had missing data")
    print(f"✓ Final missing values in numeric columns: {final_missing}")
    
    return df_filled

def run_with_custom_paths(
    msa_baseline_path,
    msa_new_data_path,
    output_path,
    date_col="Year_Month_Day",
    id_columns=["rcode", "cs_name"],
    msa_baseline_columns=[],
    target_column="HPA1Yfwd",
    msa_hpi_col="HPI",
    msa_hpa12m_col="hpa12m",
    usa_hpi_col="USA_HPI",              # USA HPI column from baseline
    usa_hpa12m_col="USA_HPA12M",        # USA HPA12M column from baseline
    usa_hpi12mF_col="USA_HPI1Yfwd",
    usa_hpa12mF_col="USA_HPA1Yfwd",
    hpi1y_fwd_col="HPI1Y_fwd",
    usa_projection_col="ProjectedHPA1YFwd_USABaseline",
    msa_projection_col="ProjectedHPA1YFwd_MSABaseline",
    msa_new_columns=[],
    start_date="1990-01-01",
    end_date="2025-01-01",
    all_models_list=None,
    all_model_params=None,
    grid_specs=None,
    title_specs=None
):
    """
    Run the MSA Calibration pipeline with custom file paths and configuration parameters.
    """
    global REGION_RESULTS
    logger.info("Running with custom paths and parameters...")

    # Set up config
    class CustomCFG:
        def __init__(self):
            self.msa_baseline_path = msa_baseline_path
            self.msa_new_data_path = msa_new_data_path
            self.output_path = output_path
            self.date_col = date_col
            self.id_columns = id_columns
            self.rcode_col = id_columns[0]  # First ID column is rcode
            self.cs_name_col = id_columns[1]  # Second ID column is cs_name
            self.msa_baseline_columns = msa_baseline_columns
            self.target_column = target_column
            self.msa_hpi_col = msa_hpi_col
            self.msa_hpa12m_col = msa_hpa12m_col
            self.usa_hpi_col = usa_hpi_col
            self.usa_hpa12m_col = usa_hpa12m_col
            self.usa_hpi12mF_col = usa_hpi12mF_col
            self.usa_hpa12mF_col = usa_hpa12mF_col
            self.hpi1y_fwd_col = hpi1y_fwd_col
            self.usa_projection_col = usa_projection_col
            self.msa_projection_col = msa_projection_col
            self.msa_new_columns = msa_new_columns
            self.start_date = start_date
            self.end_date = end_date
            self.AllModelsList = all_models_list or [
                'Ridge',
                'RandomForest',
                'XGBoost'
            ]
            self.AllModelParams = all_model_params or {
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
                'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]}
            }
            self.grid_specs = grid_specs or {
                'visible': True,
                'which': 'both',
                'linestyle': '--',
                'color': 'lightgrey',
                'linewidth': 0.75
            }
            self.title_specs = title_specs or {
                'fontsize': 9,
                'fontweight': 'bold',
                'color': '#992600',
            }
            # Additional attributes needed for data processing
            self.hpi_col = msa_hpi_col  # Alias for msa_hpi_col
            self.hpa12m_col = msa_hpa12m_col  # Alias for msa_hpa12m_col
            self.additional_features = []  # Empty list for additional features
    cfg = CustomCFG()

    # Run pipeline
    merged_df, unique_regions = loadAndMergeData(cfg)
    
    # Fill missing data by region
    merged_df = fillMissingDataByRegion(merged_df, cfg)
    
    # Create forward-looking target variables
    merged_df = createForwardLookingVariables(merged_df, cfg)
    
    merged_df = createTrainTestTags(merged_df, cfg)
    merged_df = addAllFeatures(merged_df, cfg)

    processed_count = 0
    skipped_count = 0
    REGION_RESULTS = {}
    for i, region in enumerate(unique_regions, 1):
        region_df = merged_df[merged_df[cfg.id_columns[0]] == region].copy()
        region_results = processRegionMSA(region, region_df, cfg, target_col=cfg.target_column)
        REGION_RESULTS[region] = region_results
        if region_results is not None:
            processed_count += 1
        else:
            skipped_count += 1

    if processed_count > 0:
        final_output = generateFinalOutput(merged_df, cfg)
        final_output.to_csv(cfg.output_path, index=False)
        logger.info(f"Results saved to: {cfg.output_path}")
        return final_output
    else:
        logger.error("No MSA regions were successfully processed")
        return None

def main():
    """
    Main execution function for MSA Calibration model.
    """
    global REGION_RESULTS
    
    try:
        print("="*70)
        print("MSA CALIBRATION MODEL")
        print("="*70)
        
        # Step 1: Initialize configuration
        cfg = CFG(get_user_input=False)
        
        # Step 2: Load and merge data
        merged_df, unique_regions = loadAndMergeData(cfg)
        
        # Step 2.5: Fill missing data by region
        merged_df = fillMissingDataByRegion(merged_df, cfg)
        
        # Step 3: Create forward-looking target variables
        merged_df = createForwardLookingVariables(merged_df, cfg)
        
        # Step 4: Create train/test tags
        merged_df = createTrainTestTags(merged_df, cfg)
        
        # Step 5: Add engineered features
        merged_df = addAllFeatures(merged_df, cfg)
        
        print(f"\nProcessing {len(unique_regions)} MSA regions...")
        
        # Step 6: Process each MSA region
        processed_count = 0
        skipped_count = 0
        
        for i, region in enumerate(unique_regions, 1):
            print(f"\n[{i}/{len(unique_regions)}] Processing MSA region: {region}")
            
            # Filter data for this region
            region_df = merged_df[merged_df[cfg.id_columns[0]] == region].copy()
            
            # Process the region
            region_results = processRegionMSA(region, region_df, cfg, target_col=cfg.target_column)
            
            # Store results
            REGION_RESULTS[region] = region_results
            
            if region_results is not None:
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"\n{'='*60}")
        print("MSA PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {processed_count} MSA regions")
        print(f"⚠️  Skipped: {skipped_count} MSA regions")
        print(f"📊 Total regions: {len(unique_regions)}")
        
        if processed_count > 0:
            # Step 7: Generate final output
            final_output = generateFinalOutput(merged_df, cfg)
            
            # Save results
            final_output.to_csv(cfg.output_path, index=False)
            print(f"\nResults saved to: {cfg.output_path}")
            
            print("\nSAMPLE OF FINAL OUTPUT")
            print(f"{'='*50}")
            print(final_output.head(10))
            
            print(f"\nOutput columns: {list(final_output.columns)}")
            print(f"Output shape: {final_output.shape}")
            print(f"Date range: {final_output[cfg.date_col].min()} to {final_output[cfg.date_col].max()}")
            print(f"Unique MSAs: {final_output[cfg.id_columns[0]].nunique()}")
            print(f"Train records: {(final_output['tag'] == 'train').sum()}")
            print(f"Test records: {(final_output['tag'] == 'test').sum()}")
            
        else:
            print("❌ No MSA regions were successfully processed")
        
        print(f"\n{'='*70}")
        print("MSA CALIBRATION PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()


'''
# Example usage:
run_with_custom_paths(
    msa_baseline_path='MSA_Baseline_Results.csv',  # Path to MSA baseline data
    msa_new_data_path='msa_data.csv',             # Path to new MSA data
    output_path='MSA_Calibration_Results.csv',     # Path for output file
    date_col="Year_Month_Day",                    # Date column name
    id_columns=["rcode", "cs_name"],              # ID columns to merge on
    msa_baseline_columns=[                        # Additional columns from MSA baseline to use
        "feature1",
        "feature2"
    ],
    target_column="HPA1Yfwd",                     # Target column from MSA baseline
    msa_hpi_col="HPI",                           # MSA HPI column from baseline
    msa_hpa12m_col="hpa12m",                     # MSA HPA12M column from baseline
    usa_hpi_col="USA_HPI",              # USA HPI column from baseline
    usa_hpa12m_col="USA_HPA12M",        # USA HPA12M column from baseline
    usa_hpi12mF_col="USA_HPI1Yfwd",
    usa_hpa12mF_col="USA_HPA1Yfwd",
    hpi1y_fwd_col="HPI1Y_fwd",
    usa_projection_col="ProjectedHPA1YFwd_USABaseline",  # USA projection column from baseline
    msa_projection_col="ProjectedHPA1YFwd_MSABaseline",  # MSA projection column from baseline
    msa_new_columns=[                            # Additional columns from new MSA data
        "feature1",
        "feature2"
    ],
    start_date="1990-01-01",                     # Start date
    end_date="2025-01-01"                         # End date
)
'''

# --- TESTING CODE ---
def test_pipeline_with_dummy_data():
    import pandas as pd
    # Create dummy MSA baseline data
    msa_baseline = pd.DataFrame({
        'rcode': ['A']*14 + ['B']*14,
        'cs_name': ['Alpha']*14 + ['Beta']*14,
        'Year_Month_Day': pd.date_range('2020-01-01', periods=14, freq='MS').tolist()*2,
        'HPI': list(range(100, 114)) + list(range(200, 214)),
        'hpa12m': [1.0]*14 + [2.0]*14,
        'USA_HPI1Yfwd': list(range(1000, 1014)) + list(range(2000, 2014)),
        'USA_HPA1Yfwd': [10.0]*14 + [20.0]*14,
        'HPA1Yfwd': [1.1]*12 + [None]*2 + [2.2]*12 + [None]*2,
        'ProjectedHPA1YFwd_USABaseline': [5.0]*14 + [6.0]*14,
        'ProjectedHPA1YFwd_MSABaseline': [7.0]*14 + [8.0]*14
    })
    msa_baseline.to_csv('MSA_Baseline_Results.csv', index=False)

    # Create dummy new MSA data
    msa_new = pd.DataFrame({
        'rcode': ['A']*14 + ['B']*14,
        'cs_name': ['Alpha']*14 + ['Beta']*14,
        'Year_Month_Day': pd.date_range('2020-01-01', periods=14, freq='MS').tolist()*2,
        'feature1': [0.1]*28,
        'feature2': [0.2]*28
    })
    msa_new.to_csv('msa_data.csv', index=False)

    # Run the pipeline
    run_with_custom_paths(
        msa_baseline_path='MSA_Baseline_Results.csv',
        msa_new_data_path='msa_data.csv',
        output_path='MSA_Calibration_Results.csv',
        date_col='Year_Month_Day',
        id_columns=['rcode', 'cs_name'],
        msa_baseline_columns=[],
        target_column='HPA1Yfwd',
        msa_hpi_col='HPI',
        msa_hpa12m_col='hpa12m',
        usa_hpi_col='USA_HPI',
        usa_hpa12m_col='USA_HPA12M',
        usa_hpi12mF_col='USA_HPI1Yfwd',
        usa_hpa12mF_col='USA_HPA1Yfwd',
        hpi1y_fwd_col='HPI1Y_fwd',
        usa_projection_col='ProjectedHPA1YFwd_USABaseline',
        msa_projection_col='ProjectedHPA1YFwd_MSABaseline',
        msa_new_columns=['feature1', 'feature2'],
        start_date='2020-01-01',
        end_date='2021-02-01'
    )
    print('Test pipeline run complete. Output saved to MSA_Calibration_Results.csv')

# Uncomment to run the test
test_pipeline_with_dummy_data()