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
    Configuration class for MSA Baseline model with USA data integration.
    """
    
    def __init__(self, get_user_input=True, end_date=None):
        logger.info("Initializing CFG class for MSA Baseline with USA data...")
        
        if get_user_input:
            self._get_user_configurations()
        else:
            self._set_default_configurations()
        
        # Override end_date if provided
        if end_date is not None:
            self.end_date = end_date
            logger.info(f"End date overridden to: {self.end_date}")
        
        self._setup_plotting_specs()
        logger.info("CFG initialization complete.")
    
    def _get_user_configurations(self):
        """Get all configurations from user input"""
        print("=== MSA BASELINE CONFIGURATION SETUP ===")
        
        # Data paths
        self.msa_file_path = input("Enter MSA raw data CSV file path: ")
        self.usa_file_path = input("Enter USA raw data CSV file path: ")
        self.output_path = input("Enter output CSV file path [MSA_Baseline_Results.csv]: ") or "MSA_Baseline_Results.csv"
        
        # Column configurations for MSA data
        self.date_col = input("Enter date column name [Year_Month_Day]: ") or "Year_Month_Day"
        self.rcode_col = input("Enter region code column name [rcode]: ") or "rcode"
        self.cs_name_col = input("Enter region name column name [cs_name]: ") or "cs_name"
        self.hpi_col = input("Enter HPI column name [HPI]: ") or "HPI"
        self.hpa12m_col = input("Enter HPA 12-month column name [hpa12m]: ") or "hpa12m"
        
        # Date configurations
        self.start_date = input("Enter start date (YYYY-MM-DD) [1990-01-01]: ") or "1990-01-01"
        self.end_date = input("Enter end date (YYYY-MM-DD): ")
        if not self.end_date:
            raise ValueError("❌ End date is required. Please provide a valid date in YYYY-MM-DD format.")
        
        # Feature configurations
        feature_input = input("Enter additional feature columns (comma-separated) []: ")
        self.additional_features = [f.strip() for f in feature_input.split(",")] if feature_input else []
        
        # Model configurations
        self._setup_model_configurations()
    
    def _set_default_configurations(self):
        """Set default configurations"""
        self.msa_file_path = "MSA_raw_data.csv"
        self.usa_file_path = "USA_raw_data.csv"
        self.output_path = "MSA_Baseline_Results.csv"
        
        # Column names
        self.date_col = "Year_Month_Day"
        self.rcode_col = "rcode"
        self.cs_name_col = "cs_name"
        self.hpi_col = "HPI"
        self.hpa12m_col = "hpa12m"
        
        # Date range
        self.start_date = "1990-01-01"
        self.end_date = None  # Will be set by user input or parameter
        
        # Additional features
        self.additional_features = []
        
        # Model configurations
        self._setup_model_configurations()
        
        # Lag and feature engineering settings
        self.lagList = [1,3,6,12,24]
        self.rateList = [1,3,6,12]
        self.movingAverages = [3,6,12]
        self.targetForward = 12
    
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

# Global variable to store region-wise results
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
        msa_path = Path(cfg.msa_file_path)
        usa_path = Path(cfg.usa_file_path)
        output_path = Path(cfg.output_path)
        
        # Validate input files exist
        if not msa_path.exists():
            error_msg = f"❌ MSA data file not found: {cfg.msa_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not usa_path.exists():
            error_msg = f"❌ USA data file not found: {cfg.usa_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate input files are readable
        if not os.access(msa_path, os.R_OK):
            error_msg = f"❌ MSA data file is not readable: {cfg.msa_file_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        if not os.access(usa_path, os.R_OK):
            error_msg = f"❌ USA data file is not readable: {cfg.usa_file_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        # Validate file extensions
        valid_extensions = ['.csv', '.CSV']
        if msa_path.suffix not in valid_extensions:
            error_msg = f"❌ MSA data file must be CSV format: {cfg.msa_file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if usa_path.suffix not in valid_extensions:
            error_msg = f"❌ USA data file must be CSV format: {cfg.usa_file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check file sizes (warn if very large)
        msa_size_mb = msa_path.stat().st_size / (1024 * 1024)
        usa_size_mb = usa_path.stat().st_size / (1024 * 1024)
        
        if msa_size_mb > 500:  # 500 MB
            logger.warning(f"⚠️  Large MSA file detected: {msa_size_mb:.1f} MB. Processing may take longer.")
            print(f"⚠️  Large MSA file detected: {msa_size_mb:.1f} MB. Processing may take longer.")
        
        if usa_size_mb > 100:  # 100 MB
            logger.warning(f"⚠️  Large USA file detected: {usa_size_mb:.1f} MB. Processing may take longer.")
            print(f"⚠️  Large USA file detected: {usa_size_mb:.1f} MB. Processing may take longer.")
        
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
        cfg.msa_file_path = str(msa_path.resolve())
        cfg.usa_file_path = str(usa_path.resolve())
        cfg.output_path = str(output_path.resolve())
        
        logger.info("✓ All file paths validated successfully")
        print("✓ All file paths validated successfully")
        print(f"  MSA data: {cfg.msa_file_path} ({msa_size_mb:.1f} MB)")
        print(f"  USA data: {cfg.usa_file_path} ({usa_size_mb:.1f} MB)")
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
            error_msg = "❌ MSA data file is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if usa_df.empty:
            error_msg = "❌ USA data file is empty"  
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check required columns exist in MSA data
        required_msa_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col, cfg.hpi_col, cfg.hpa12m_col]
        missing_msa_cols = [col for col in required_msa_cols if col not in msa_df.columns]
        
        if missing_msa_cols:
            error_msg = f"❌ Missing required columns in MSA data: {missing_msa_cols}"
            logger.error(error_msg)
            print(f"Available MSA columns: {list(msa_df.columns)}")
            raise ValueError(error_msg)
        
        # Check required columns exist in USA data
        required_usa_cols = [cfg.date_col]
        missing_usa_cols = [col for col in required_usa_cols if col not in usa_df.columns]
        
        if missing_usa_cols:
            error_msg = f"❌ Missing required columns in USA data: {missing_usa_cols}"
            logger.error(error_msg)
            print(f"Available USA columns: {list(usa_df.columns)}")
            raise ValueError(error_msg)
        
        # Check date column format
        try:
            pd.to_datetime(msa_df[cfg.date_col])
        except Exception:
            error_msg = f"❌ Invalid date format in MSA data column '{cfg.date_col}'"
            logger.error(error_msg)
            print(f"Sample MSA date values: {msa_df[cfg.date_col].head().tolist()}")
            raise ValueError(error_msg)
        
        try:
            pd.to_datetime(usa_df[cfg.date_col])
        except Exception:
            error_msg = f"❌ Invalid date format in USA data column '{cfg.date_col}'"
            logger.error(error_msg)
            print(f"Sample USA date values: {usa_df[cfg.date_col].head().tolist()}")
            raise ValueError(error_msg)
        
        # Check for minimum data requirements
        if len(msa_df) < 12:
            logger.warning(f"⚠️  MSA data has very few records ({len(msa_df)}). Results may be unreliable.")
            print(f"⚠️  MSA data has very few records ({len(msa_df)}). Results may be unreliable.")
        
        if len(usa_df) < 12:
            logger.warning(f"⚠️  USA data has very few records ({len(usa_df)}). Results may be unreliable.")
            print(f"⚠️  USA data has very few records ({len(usa_df)}). Results may be unreliable.")
        
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
        print(f"  MSA records: {total_msa_records}")
        print(f"  USA records: {len(usa_df)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data integrity validation failed: {str(e)}")
        print(f"❌ Data integrity validation failed: {str(e)}")
        raise e

def loadAndMergeData(cfg):
    """
    Load MSA and USA data, then merge them.
    
    Parameters:
    - cfg: Configuration object
    
    Returns:
    - merged_df: Merged dataframe with both MSA and USA data
    - unique_regions: List of unique MSA regions
    """
    logger.info("Step 2: Loading and merging MSA and USA data...")
    print("Loading MSA and USA raw data...")
    
    try:
        # Load MSA data with error handling
        print(f"Loading MSA data from: {cfg.msa_file_path}")
        try:
            msa_df = pd.read_csv(cfg.msa_file_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"❌ MSA data file is empty: {cfg.msa_file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"❌ Error parsing MSA data file: {cfg.msa_file_path}. {str(e)}")
        except UnicodeDecodeError:
            # Try different encodings
            try:
                msa_df = pd.read_csv(cfg.msa_file_path, encoding='latin-1')
                logger.warning("⚠️  MSA file loaded with latin-1 encoding")
            except:
                raise ValueError(f"❌ Cannot read MSA data file due to encoding issues: {cfg.msa_file_path}")
        
        logger.info(f"MSA data loaded. Shape: {msa_df.shape}")
        print(f"MSA data shape: {msa_df.shape}")
        
        # Load USA data with error handling
        print(f"Loading USA data from: {cfg.usa_file_path}")
        try:
            usa_df = pd.read_csv(cfg.usa_file_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"❌ USA data file is empty: {cfg.usa_file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"❌ Error parsing USA data file: {cfg.usa_file_path}. {str(e)}")
        except UnicodeDecodeError:
            # Try different encodings
            try:
                usa_df = pd.read_csv(cfg.usa_file_path, encoding='latin-1')
                logger.warning("⚠️  USA file loaded with latin-1 encoding")
            except:
                raise ValueError(f"❌ Cannot read USA data file due to encoding issues: {cfg.usa_file_path}")
        
        logger.info(f"USA data loaded. Shape: {usa_df.shape}")
        print(f"USA data shape: {usa_df.shape}")
        
        # Validate data integrity
        validateDataIntegrity(msa_df, usa_df, cfg)
        
        # Convert date columns to datetime
        msa_df[cfg.date_col] = pd.to_datetime(msa_df[cfg.date_col])
        usa_df[cfg.date_col] = pd.to_datetime(usa_df[cfg.date_col])
        
        # Filter MSA data by date range
        start_dt = pd.to_datetime(cfg.start_date)
        end_dt = pd.to_datetime(cfg.end_date)
        
        msa_df_filtered = msa_df[(msa_df[cfg.date_col] >= start_dt) & (msa_df[cfg.date_col] <= end_dt)].copy()
        usa_df_filtered = usa_df[(usa_df[cfg.date_col] >= start_dt) & (usa_df[cfg.date_col] <= end_dt)].copy()
        
        logger.info(f"MSA data filtered. Shape: {msa_df_filtered.shape}")
        logger.info(f"USA data filtered. Shape: {usa_df_filtered.shape}")
        
        # Prepare USA data for merging (rename columns to avoid conflicts)
        usa_merge_cols = [cfg.date_col]
        if 'ProjectedHPA1YFwd_USABaseline' in usa_df_filtered.columns:
            usa_merge_cols.append('ProjectedHPA1YFwd_USABaseline')
        if 'USA_HPA1Yfwd' in usa_df_filtered.columns:
            usa_merge_cols.append('USA_HPA1Yfwd')
        if 'USA_HPI1Yfwd' in usa_df_filtered.columns:
            usa_merge_cols.append('USA_HPI1Yfwd')
        
        usa_for_merge = usa_df_filtered[usa_merge_cols].copy()
        
        # Merge MSA and USA data on date
        print("Merging MSA and USA data...")
        merged_df = msa_df_filtered.merge(usa_for_merge, on=cfg.date_col, how='left')
        
        logger.info(f"Data merged successfully. Shape: {merged_df.shape}")
        print(f"Merged data shape: {merged_df.shape}")
        
        # Get unique regions
        unique_regions = merged_df[cfg.rcode_col].unique()
        unique_regions = [r for r in unique_regions if pd.notna(r)]
        
        logger.info(f"Found {len(unique_regions)} unique MSA regions")
        print(f"Found {len(unique_regions)} unique MSA regions")
        
        return merged_df, unique_regions
        
    except Exception as e:
        logger.error(f"Error in loadAndMergeData: {str(e)}")
        raise e

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
    logger.info("Step 1.5: Filling missing data by region...")
    print("Filling missing data using growth/decay rates and monthly averages...")
    
    df_filled = df.copy()
    
    # Define numeric columns to fill (exclude ID and date columns)
    exclude_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col]
    numeric_cols = [col for col in df_filled.columns 
                   if col not in exclude_cols and df_filled[col].dtype in ['float64', 'int64']]
    
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
            missing_count = region_df[col].isnull().sum()
            
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
                    region_series = region_df[col].copy()
                    
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

def createForwardLookingVariables(df, cfg):
    """
    Create forward-looking variables (HPA1Yfwd, HPI1Y_fwd) for each MSA.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Dataframe with forward-looking variables
    """
    logger.info("Step 3: Creating forward-looking variables...")
    print("Creating 1-year forward variables...")
    
    df_enhanced = df.copy()
    
    # Sort by region and date
    df_enhanced = df_enhanced.sort_values([cfg.rcode_col, cfg.date_col])
    
    # Create HPA1Yfwd (1 year forward HPA)
    df_enhanced['HPA1Yfwd'] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[cfg.hpa12m_col].shift(-12)
    
    # Create HPI1Y_fwd (1 year forward HPI)
    df_enhanced['HPI1Y_fwd'] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[cfg.hpi_col].shift(-12)
    
    logger.info("Forward-looking variables created successfully")
    print("✓ Forward-looking variables created")
    
    return df_enhanced

def createTrainTestTags(df, cfg):
    """
    Create train/test tags. Test data is the last year where we don't have forward-looking data.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Dataframe with train/test tags
    """
    logger.info("Step 4: Creating train/test tags...")
    print("Creating train/test split tags...")
    
    df_tagged = df.copy()
    
    # Initialize tag column
    df_tagged['tag'] = 'Train'
    
    # For each MSA, mark the last 12 months (where HPA1Yfwd is NaN) as test
    for region in df_tagged[cfg.rcode_col].unique():
        if pd.isna(region):
            continue
            
        region_mask = df_tagged[cfg.rcode_col] == region
        region_df = df_tagged[region_mask].copy()
        
        # Find rows where HPA1Yfwd is NaN (these are test rows)
        test_mask = region_df['HPA1Yfwd'].isna()
        
        # Update the main dataframe
        df_tagged.loc[region_mask & df_tagged['HPA1Yfwd'].isna(), 'tag'] = 'Test'
    
    train_count = (df_tagged['tag'] == 'Train').sum()
    test_count = (df_tagged['tag'] == 'Test').sum()
    
    logger.info(f"Train/test tags created. Train: {train_count}, Test: {test_count}")
    print(f"✓ Train/test split created - Train: {train_count}, Test: {test_count}")
    
    return df_tagged

def addAllFeatures(df, cfg):
    """
    Add all lagged features, moving averages, and other engineered features using hpa12m as base.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Enhanced dataframe with new features
    """
    logger.info("Step 5: Adding engineered features...")
    print("Adding comprehensive feature set...")
    
    df_enhanced = df.copy()
    
    # Define features to engineer (hpa12m is the main target feature)
    base_features = [cfg.hpa12m_col, cfg.hpi_col]
    
    # Add USA baseline features if available
    if 'ProjectedHPA1YFwd_USABaseline' in df_enhanced.columns:
        base_features.append('ProjectedHPA1YFwd_USABaseline')
    if 'USA_HPA1Yfwd' in df_enhanced.columns:
        base_features.append('USA_HPA1Yfwd')
    if 'USA_HPI1Yfwd' in df_enhanced.columns:
        base_features.append('USA_HPI1Yfwd')
    
    # Add any additional features from config
    base_features.extend(cfg.additional_features)
    
    # Remove duplicates and ensure columns exist
    base_features = list(set([f for f in base_features if f in df_enhanced.columns]))
    
    # Sort by region and date for proper lag calculation
    df_enhanced = df_enhanced.sort_values([cfg.rcode_col, cfg.date_col])
    
    for feature in base_features:
        print(f"Processing feature: {feature}")
        
        # 1. Lagged features
        for lag in cfg.lagList:
            lag_col = f"{feature}_lag_{lag}"
            df_enhanced[lag_col] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[feature].shift(lag)
        
        # 2. Moving averages
        for ma in cfg.movingAverages:
            ma_col = f"{feature}_ma_{ma}"
            df_enhanced[ma_col] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[feature].rolling(
                window=ma, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        
        # 3. Rate of change
        for rate in cfg.rateList:
            rate_col = f"{feature}_pct_{rate}m"
            df_enhanced[rate_col] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[feature].pct_change(periods=rate)
        
        # 4. Min/Max in last 12 months
        min_col = f"{feature}_min_12m"
        max_col = f"{feature}_max_12m"
        df_enhanced[min_col] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[feature].rolling(
            window=12, min_periods=1).min().reset_index(level=[0,1], drop=True)
        df_enhanced[max_col] = df_enhanced.groupby([cfg.rcode_col, cfg.cs_name_col])[feature].rolling(
            window=12, min_periods=1).max().reset_index(level=[0,1], drop=True)
    
    logger.info(f"Feature engineering complete. New shape: {df_enhanced.shape}")
    print(f"✓ Feature engineering complete. DataFrame shape: {df_enhanced.shape}")
    
    return df_enhanced

def processRegionMSA(region, df_region, cfg, target_col='HPA1Yfwd'):
    """
    Process a single MSA region through the modeling pipeline.
    
    Parameters:
    - region: MSA region identifier (rcode)
    - df_region: Data for specific MSA region
    - cfg: Configuration object
    - target_col: Target column name
    
    Returns:
    - region_results: Dictionary containing all results for the region
    """
    logger.info(f"Processing MSA region: {region}")
    print(f"\n{'='*60}")
    print(f"PROCESSING MSA REGION: {region}")
    print(f"{'='*60}")
    
    try:
        # Check if region has sufficient data
        if df_region.shape[0] < 24:  # At least 2 years of data
            logger.warning(f"Region {region} has insufficient data ({df_region.shape[0]} rows), skipping...")
            print(f"⚠️  Skipping {region}: insufficient data ({df_region.shape[0]} rows)")
            return None
        
        # Separate train and test data
        train_df = df_region[df_region['tag'] == 'Train'].copy()
        test_df = df_region[df_region['tag'] == 'Test'].copy()
        
        # Check if we have enough training data
        if len(train_df) < 12:  # At least 1 year of training data
            logger.warning(f"Region {region} has insufficient training data ({len(train_df)} rows), skipping...")
            print(f"⚠️  Skipping {region}: insufficient training data ({len(train_df)} rows)")
            return None
        
        # Get feature columns (exclude ID, date, target, and tag columns)
        exclude_cols = [cfg.date_col, cfg.rcode_col, cfg.cs_name_col, 'tag', target_col, 
                       'HPA1Yfwd', 'HPI1Y_fwd', cfg.hpa12m_col, cfg.hpi_col]
        x_columns = [col for col in train_df.columns if col not in exclude_cols and not train_df[col].isnull().all()]
        
        # Check if we have target values in training data
        if target_col not in train_df.columns or train_df[target_col].isnull().all():
            logger.warning(f"Region {region} has no valid target values, skipping...")
            print(f"⚠️  Skipping {region}: no valid target values")
            return None
        
        # Remove rows with NaN target values from training data
        train_df = train_df.dropna(subset=[target_col])
        
        if len(train_df) < 12:
            logger.warning(f"Region {region} has insufficient training data after removing NaN targets, skipping...")
            print(f"⚠️  Skipping {region}: insufficient training data after cleaning")
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
            print(f"⚠️  Skipping {region}: too few features ({len(x_columns)})")
            return None
        
        # Standardize features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Simple model selection (use fewer models for efficiency)
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
            print(f"⚠️  No models trained successfully for {region}")
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
        
        print(f"✓ Region {region} processed successfully")
        print(f"✓ Training R²: {train_r2:.4f}")
        print(f"✓ Features used: {len(x_columns)}")
        print(f"✓ Training samples: {len(train_df)}")
        print(f"✓ Test samples: {len(test_df)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing region {region}: {str(e)}")
        print(f"❌ Error processing region {region}: {str(e)}")
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
    logger.info("Step 7: Generating final output...")
    print("Generating final output with all required columns...")
    
    # Start with the base dataframe
    final_df = merged_df.copy()
    
    # Initialize ProjectedHPA1YFwd_MSABaseline column
    final_df['ProjectedHPA1YFwd_MSABaseline'] = np.nan
    
    # Fill in MSA baseline projections from region results
    for region, results in REGION_RESULTS.items():
        if results is None:
            continue
        
        try:
            # Get region mask
            region_mask = final_df[cfg.rcode_col] == region
            
            # Fill training predictions
            train_indices = results['train_data'].index
            for idx, pred in zip(train_indices, results['train_predictions']):
                if idx in final_df.index:
                    final_df.loc[idx, 'ProjectedHPA1YFwd_MSABaseline'] = pred
            
            # Fill test predictions
            test_indices = results['test_data'].index
            for idx, pred in zip(test_indices, results['test_predictions']):
                if idx in final_df.index:
                    final_df.loc[idx, 'ProjectedHPA1YFwd_MSABaseline'] = pred
                    
        except Exception as e:
            logger.warning(f"Error filling predictions for region {region}: {str(e)}")
    
    # Ensure USA baseline columns are present (fill with NaN if missing)
    if 'ProjectedHPA1YFwd_USABaseline' not in final_df.columns:
        final_df['ProjectedHPA1YFwd_USABaseline'] = np.nan
    if 'USA_HPA1Yfwd' not in final_df.columns:
        final_df['USA_HPA1Yfwd'] = np.nan
    if 'USA_HPI1Yfwd' not in final_df.columns:
        final_df['USA_HPI1Yfwd'] = np.nan
    
    # Set USA baseline columns to NaN for test rows
    test_mask = final_df['tag'] == 'Test'
    final_df.loc[test_mask, 'USA_HPA1Yfwd'] = np.nan
    final_df.loc[test_mask, 'USA_HPI1Yfwd'] = np.nan
    
    # Select and order the required columns
    required_columns = [
        cfg.date_col,  # 'Year_Month_Day'
        cfg.rcode_col,  # 'rcode'
        cfg.cs_name_col,  # 'cs_name'
        'tag',
        'ProjectedHPA1YFwd_USABaseline',
        'ProjectedHPA1YFwd_MSABaseline',
        cfg.hpi_col,  # 'HPI'
        cfg.hpa12m_col,  # 'hpa12m'
        'HPA1Yfwd',
        'HPI1Y_fwd',
        'USA_HPA1Yfwd',
        'USA_HPI1Yfwd'
    ]
    
    # Rename date column to match required output
    if cfg.date_col != 'Year_Month_Day':
        final_df = final_df.rename(columns={cfg.date_col: 'Year_Month_Day'})
        required_columns[0] = 'Year_Month_Day'
    
    # Select only the required columns
    final_df = final_df[required_columns]
    
    # Convert date to YYYY-MM-01 format
    final_df['Year_Month_Day'] = pd.to_datetime(final_df['Year_Month_Day']).dt.to_period('M').dt.start_time
    
    # Sort by region and date
    final_df = final_df.sort_values([cfg.rcode_col, 'Year_Month_Day'])
    
    logger.info(f"Final output generated. Shape: {final_df.shape}")
    print(f"✓ Final output generated. Shape: {final_df.shape}")
    
    return final_df

# Main execution function
def main():
    """
    Main execution function for MSA Baseline model.
    """
    global REGION_RESULTS
    
    try:
        print("="*70)
        print("MSA BASELINE MODEL WITH USA DATA INTEGRATION")
        print("="*70)
        
        # Step 1: Initialize configuration
        print("="*70)
        print("MSA BASELINE CONFIGURATION")
        print("="*70)
        
        # Get end date from user
        end_date = input("Enter end date (YYYY-MM-DD): ")
        if not end_date:
            raise ValueError("❌ End date is required. Please provide a valid date in YYYY-MM-DD format.")
        
        # Validate date format
        try:
            pd.to_datetime(end_date)
        except:
            raise ValueError("❌ Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-12-31)")
        
        cfg = CFG(get_user_input=False, end_date=end_date)
        
        # Step 1.5: Validate file paths and prepare directories
        validateAndPreparePaths(cfg)
        
        # Step 2: Load and merge data
        merged_df, unique_regions = loadAndMergeData(cfg)
        
        # Step 2.5: Fill missing data by region
        merged_df = fillMissingDataByRegion(merged_df, cfg)
        
        # Step 3: Create forward-looking variables
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
            region_df = merged_df[merged_df[cfg.rcode_col] == region].copy()
            
            # Process the region
            region_results = processRegionMSA(region, region_df, cfg, target_col='HPA1Yfwd')
            
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
        
        # Step 7: Generate final output
        if processed_count > 0:
            final_output = generateFinalOutput(merged_df, cfg)
            
            # Save final output with error handling
            try:
                print(f"Saving final output to: {cfg.output_path}")
                final_output.to_csv(cfg.output_path, index=False)
                
                # Verify the file was saved successfully
                if os.path.exists(cfg.output_path):
                    file_size = os.path.getsize(cfg.output_path)
                    logger.info(f"Final output saved successfully to: {cfg.output_path} ({file_size} bytes)")
                    print(f"✓ Final output saved to: {cfg.output_path} ({file_size / 1024:.1f} KB)")
                else:
                    raise FileNotFoundError("Output file was not created successfully")
                    
            except PermissionError:
                error_msg = f"❌ Permission denied: Cannot write to {cfg.output_path}"
                logger.error(error_msg)
                print(error_msg)
                print("  Try running with administrator privileges or choose a different output location.")
                raise
            except OSError as e:
                error_msg = f"❌ OS Error writing file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            except Exception as e:
                error_msg = f"❌ Unexpected error saving file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            
            # Display sample of final output
            print(f"\n{'='*50}")
            print("SAMPLE OF FINAL OUTPUT")
            print(f"{'='*50}")
            print(final_output.head(10))
            
            print(f"\nOutput columns: {list(final_output.columns)}")
            print(f"Output shape: {final_output.shape}")
            print(f"Date range: {final_output['Year_Month_Day'].min()} to {final_output['Year_Month_Day'].max()}")
            print(f"Unique MSAs: {final_output['rcode'].nunique()}")
            print(f"Train records: {(final_output['tag'] == 'Train').sum()}")
            print(f"Test records: {(final_output['tag'] == 'Test').sum()}")
            
        else:
            print("❌ No MSA regions were successfully processed")
        
        print(f"\n{'='*70}")
        print("MSA BASELINE PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

# Example usage with custom paths and model parameters
def run_with_custom_paths(msa_file, usa_file, output_file, end_date=None, model_params=None):
    """
    Run the MSA baseline model with custom file paths and model parameters.
    
    Parameters:
    - msa_file: Path to MSA raw data CSV
    - usa_file: Path to USA raw data CSV  
    - output_file: Path for output CSV
    - end_date: End date in YYYY-MM-DD format (required)
    - model_params: Dictionary of model parameters to override defaults
    """
    global REGION_RESULTS
    REGION_RESULTS = {}  # Reset results
    
    try:
        # Validate input parameters
        if not msa_file or not isinstance(msa_file, str):
            raise ValueError("❌ MSA file path must be a non-empty string")
        if not usa_file or not isinstance(usa_file, str):
            raise ValueError("❌ USA file path must be a non-empty string")
        if not output_file or not isinstance(output_file, str):
            raise ValueError("❌ Output file path must be a non-empty string")
        if not end_date or not isinstance(end_date, str):
            raise ValueError("❌ End date must be a non-empty string in YYYY-MM-DD format")
        
        # Validate end date format
        try:
            pd.to_datetime(end_date)
        except:
            raise ValueError("❌ Invalid end date format. Please use YYYY-MM-DD format (e.g., 2024-12-31)")
        
        # Create configuration with custom paths
        cfg = CFG(get_user_input=False, end_date=end_date)
        cfg.msa_file_path = msa_file
        cfg.usa_file_path = usa_file
        cfg.output_path = output_file
        
        # Update model parameters if provided
        if model_params:
            if not isinstance(model_params, dict):
                raise ValueError("❌ model_params must be a dictionary")
            
            # Update model list if provided
            if 'models' in model_params:
                cfg.AllModelsList = model_params['models']
            
            # Update model parameters if provided
            if 'model_params' in model_params:
                for model_name, params in model_params['model_params'].items():
                    if model_name in cfg.AllModelParams:
                        cfg.AllModelParams[model_name].update(params)
            
            # Update feature engineering parameters if provided
            if 'lag_list' in model_params:
                cfg.lagList = model_params['lag_list']
            if 'rate_list' in model_params:
                cfg.rateList = model_params['rate_list']
            if 'moving_averages' in model_params:
                cfg.movingAverages = model_params['moving_averages']
            if 'target_forward' in model_params:
                cfg.targetForward = model_params['target_forward']
        
        print(f"Running MSA Baseline with custom paths:")
        print(f"  MSA data: {msa_file}")
        print(f"  USA data: {usa_file}")
        print(f"  Output: {output_file}")
        print("\nModel Configuration:")
        print(f"  Models: {cfg.AllModelsList}")
        print(f"  Lags: {cfg.lagList}")
        print(f"  Rates: {cfg.rateList}")
        print(f"  Moving Averages: {cfg.movingAverages}")
        print(f"  Target Forward: {cfg.targetForward}")
        
        # Validate paths before running
        validateAndPreparePaths(cfg)
        
        # Run the main pipeline (skip the initial validation in main since we did it here)
        print("="*70)
        print("MSA BASELINE MODEL WITH USA DATA INTEGRATION")
        print("="*70)
        
        # Load and merge data
        merged_df, unique_regions = loadAndMergeData(cfg)
        
        # Fill missing data by region
        merged_df = fillMissingDataByRegion(merged_df, cfg)
        
        # Create forward-looking variables
        merged_df = createForwardLookingVariables(merged_df, cfg)
        
        # Create train/test tags
        merged_df = createTrainTestTags(merged_df, cfg)
        
        # Add engineered features
        merged_df = addAllFeatures(merged_df, cfg)
        
        print(f"\nProcessing {len(unique_regions)} MSA regions...")
        
        # Process each MSA region
        processed_count = 0
        skipped_count = 0
        
        for i, region in enumerate(unique_regions, 1):
            print(f"\n[{i}/{len(unique_regions)}] Processing MSA region: {region}")
            
            # Filter data for this region
            region_df = merged_df[merged_df[cfg.rcode_col] == region].copy()
            
            # Process the region
            region_results = processRegionMSA(region, region_df, cfg, target_col='HPA1Yfwd')
            
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
        print(f"📊 Total regions processed: {len(unique_regions)}")
        
        # Generate final output
        if processed_count > 0:
            final_output = generateFinalOutput(merged_df, cfg)
            
            # Save final output with error handling
            try:
                print(f"Saving final output to: {cfg.output_path}")
                final_output.to_csv(cfg.output_path, index=False)
                
                # Verify the file was saved successfully
                if os.path.exists(cfg.output_path):
                    file_size = os.path.getsize(cfg.output_path)
                    logger.info(f"Final output saved successfully to: {cfg.output_path} ({file_size} bytes)")
                    print(f"✓ Final output saved to: {cfg.output_path} ({file_size / 1024:.1f} KB)")
                else:
                    raise FileNotFoundError("Output file was not created successfully")
                    
            except PermissionError:
                error_msg = f"❌ Permission denied: Cannot write to {cfg.output_path}"
                logger.error(error_msg)
                print(error_msg)
                print("  Try running with administrator privileges or choose a different output location.")
                raise
            except OSError as e:
                error_msg = f"❌ OS Error writing file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            except Exception as e:
                error_msg = f"❌ Unexpected error saving file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            
            # Display sample of final output
            print(f"\n{'='*50}")
            print("SAMPLE OF FINAL OUTPUT")
            print(f"{'='*50}")
            print(final_output.head(10))
            
            print(f"\nOutput columns: {list(final_output.columns)}")
            print(f"Output shape: {final_output.shape}")
            print(f"Date range: {final_output['Year_Month_Day'].min()} to {final_output['Year_Month_Day'].max()}")
            print(f"Unique MSAs: {final_output['rcode'].nunique()}")
            print(f"Train records: {(final_output['tag'] == 'Train').sum()}")
            print(f"Test records: {(final_output['tag'] == 'Test').sum()}")
            
        else:
            print("❌ No MSA regions were successfully processed")
        
        print(f"\n{'='*70}")
        print("MSA BASELINE PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Custom paths pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        raise e

# Example usage with model parameters
def run_with_model_params():
    """
    Example of running the MSA baseline with custom model parameters.
    """
    model_params = {
        'models': ['Ridge', 'RandomForest', 'XGBoost'],
        'model_params': {
            'Ridge': {'alpha': [0.1, 1.0, 10.0]},
            'RandomForest': {'n_estimators': [200], 'max_depth': [20]},
            'XGBoost': {'n_estimators': [200], 'max_depth': [10], 'learning_rate': [0.1]}
        },
        'lag_list': [1, 3, 6, 12],
        'rate_list': [1, 3, 6],
        'moving_averages': [3, 6, 12],
        'target_forward': 12
    }
    
    run_with_custom_paths(
        msa_file='sample_msa_data.csv',
        usa_file='sample_usa_data.csv',
        output_file='sample_output.csv',
        end_date='2023-12-31',
        model_params=model_params
    )

# Uncomment to run with custom model parameters
# run_with_model_params()

# def create_sample_data_demo():
#     """
#     Create sample data to demonstrate the MSA baseline script functionality.
#     """
#     print("Creating sample MSA and USA data for demonstration...")
#     
#     # Create sample dates
#     dates = pd.date_range(start='2020-01-01', end='2023-12-01', freq='MS')
#     
#     # Sample MSA data
#     msa_regions = ['MSA_12345', 'MSA_23456', 'MSA_34567']
#     region_names = ['Metro Area 1', 'Metro Area 2', 'Metro Area 3']
#     
#     msa_data = []
#     for region, name in zip(msa_regions, region_names):
#         for date in dates:
#             # Simulate realistic housing data
#             base_hpi = 250 + np.random.uniform(-50, 50)
#             seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * (date.month - 1) / 12)
#             trend_factor = 1 + 0.03 * (date.year - 2020)
#             
#             hpi = base_hpi * seasonal_factor * trend_factor + np.random.normal(0, 5)
#             hpa12m = np.random.uniform(0.02, 0.08) + 0.01 * np.sin(2 * np.pi * (date.month - 1) / 12)
#             
#             msa_data.append({
#                 'Year_Month_Day': date,
#                 'rcode': region,
#                 'cs_name': name,
#                 'HPI': hpi,
#                 'hpa12m': hpa12m,
#                 'unemployment_rate': np.random.uniform(3, 7),
#                 'median_income': np.random.uniform(50000, 80000)
#             })
#     
#     msa_df = pd.DataFrame(msa_data)
#     
#     # Sample USA data
#     usa_data = []
#     for date in dates:
#         usa_data.append({
#             'Year_Month_Day': date,
#             'ProjectedHPA1YFwd_USABaseline': np.random.uniform(0.03, 0.07),
#             'USA_HPA1Yfwd': np.random.uniform(0.025, 0.065),
#             'USA_HPI1Yfwd': 260 + np.random.uniform(-10, 10)
#         })
#     
#     usa_df = pd.DataFrame(usa_data)
#     
#     # Save sample data
#     msa_df.to_csv('sample_msa_data.csv', index=False)
#     usa_df.to_csv('sample_usa_data.csv', index=False)
#     
#     print(f"✓ Sample MSA data saved: sample_msa_data.csv ({msa_df.shape})")
#     print(f"✓ Sample USA data saved: sample_usa_data.csv ({usa_df.shape})")
#     print("✓ You can now run: run_with_custom_paths('sample_msa_data.csv', 'sample_usa_data.csv', 'sample_output.csv', '2023-12-31')")
#     
#     return msa_df, usa_df

# Uncomment the lines below to create and test with sample data
# create_sample_data_demo()
# run_with_custom_paths('sample_msa_data.csv', 'sample_usa_data.csv', 'sample_output.csv', '2023-12-31')

'''
# ============================================================================
# EXAMPLE USAGE CODE - UNCOMMENT TO USE
# ============================================================================

# Example 1: Run with custom paths and model parameters
model_params = {
    'models': ['Ridge', 'RandomForest', 'XGBoost'],
    'model_params': {
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'RandomForest': {'n_estimators': [200], 'max_depth': [20]},
        'XGBoost': {'n_estimators': [200], 'max_depth': [10], 'learning_rate': [0.1]}
    },
    'lag_list': [1, 3, 6, 12],
    'rate_list': [1, 3, 6],
    'moving_averages': [3, 6, 12],
    'target_forward': 12
}

run_with_custom_paths(
    msa_file='your_msa_data.csv',
    usa_file='your_usa_data.csv',
    output_file='your_output.csv',
    end_date='2024-12-31',
    model_params=model_params
)

# Example 2: Run test with specific rcodes
test_rcodes = ['MSA_12345', 'MSA_23456', 'MSA_34567']

test_model_params = {
    'models': ['Ridge', 'RandomForest', 'XGBoost'],
    'model_params': {
        'Ridge': {'alpha': [1.0]},
        'RandomForest': {'n_estimators': [100], 'max_depth': [10]},
        'XGBoost': {'n_estimators': [100], 'max_depth': [6], 'learning_rate': [0.1]}
    },
    'lag_list': [1, 3, 6, 12],
    'rate_list': [1, 3, 6],
    'moving_averages': [3, 6, 12],
    'target_forward': 12
}

run_test_with_rcodes(
    rcodes_list=test_rcodes,
    msa_file='your_msa_data.csv',
    usa_file='your_usa_data.csv',
    output_file='testBaseline_Results.csv',
    end_date='2024-12-31',
    model_params=test_model_params
)

# Example 3: Simple run with minimal parameters
run_with_custom_paths(
    msa_file='MSA_raw_data.csv',
    usa_file='USA_raw_data.csv',
    output_file='MSA_Baseline_Results.csv',
    end_date='2024-12-31'
)

# ============================================================================
# END EXAMPLE USAGE CODE
# ============================================================================
'''

def run_test_with_rcodes(rcodes_list, msa_file, usa_file, output_file, end_date=None, model_params=None):
    """
    Run the MSA baseline model with custom file paths and model parameters for specific rcodes only.
    This is useful for testing the pipeline on a subset of regions.
    
    Parameters:
    - rcodes_list: List of rcodes to process (e.g., ['MSA_12345', 'MSA_23456'])
    - msa_file: Path to MSA raw data CSV
    - usa_file: Path to USA raw data CSV  
    - output_file: Path for output CSV
    - end_date: End date in YYYY-MM-DD format (required)
    - model_params: Dictionary of model parameters to override defaults
    """
    global REGION_RESULTS
    REGION_RESULTS = {}  # Reset results
    
    try:
        # Validate input parameters
        if not rcodes_list or not isinstance(rcodes_list, list) or len(rcodes_list) == 0:
            raise ValueError("❌ rcodes_list must be a non-empty list of rcodes")
        if not msa_file or not isinstance(msa_file, str):
            raise ValueError("❌ MSA file path must be a non-empty string")
        if not usa_file or not isinstance(usa_file, str):
            raise ValueError("❌ USA file path must be a non-empty string")
        if not output_file or not isinstance(output_file, str):
            raise ValueError("❌ Output file path must be a non-empty string")
        if not end_date or not isinstance(end_date, str):
            raise ValueError("❌ End date must be a non-empty string in YYYY-MM-DD format")
        
        # Validate end date format
        try:
            pd.to_datetime(end_date)
        except:
            raise ValueError("❌ Invalid end date format. Please use YYYY-MM-DD format (e.g., 2024-12-31)")
        
        # Create configuration with custom paths
        cfg = CFG(get_user_input=False, end_date=end_date)
        cfg.msa_file_path = msa_file
        cfg.usa_file_path = usa_file
        cfg.output_path = output_file
        
        # Update model parameters if provided
        if model_params:
            if not isinstance(model_params, dict):
                raise ValueError("❌ model_params must be a dictionary")
            
            # Update model list if provided
            if 'models' in model_params:
                cfg.AllModelsList = model_params['models']
            
            # Update model parameters if provided
            if 'model_params' in model_params:
                for model_name, params in model_params['model_params'].items():
                    if model_name in cfg.AllModelParams:
                        cfg.AllModelParams[model_name].update(params)
            
            # Update feature engineering parameters if provided
            if 'lag_list' in model_params:
                cfg.lagList = model_params['lag_list']
            if 'rate_list' in model_params:
                cfg.rateList = model_params['rate_list']
            if 'moving_averages' in model_params:
                cfg.movingAverages = model_params['moving_averages']
            if 'target_forward' in model_params:
                cfg.targetForward = model_params['target_forward']
        
        print(f"Running MSA Baseline TEST with specific rcodes:")
        print(f"  Target rcodes: {rcodes_list}")
        print(f"  MSA data: {msa_file}")
        print(f"  USA data: {usa_file}")
        print(f"  Output: {output_file}")
        print("\nModel Configuration:")
        print(f"  Models: {cfg.AllModelsList}")
        print(f"  Lags: {cfg.lagList}")
        print(f"  Rates: {cfg.rateList}")
        print(f"  Moving Averages: {cfg.movingAverages}")
        print(f"  Target Forward: {cfg.targetForward}")
        
        # Validate paths before running
        validateAndPreparePaths(cfg)
        
        # Run the main pipeline (skip the initial validation in main since we did it here)
        print("="*70)
        print("MSA BASELINE TEST WITH SPECIFIC RCODES")
        print("="*70)
        
        # Load and merge data
        merged_df, unique_regions = loadAndMergeData(cfg)
        
        # Filter data to only include the specified rcodes
        print(f"Filtering data to include only specified rcodes: {rcodes_list}")
        merged_df = merged_df[merged_df[cfg.rcode_col].isin(rcodes_list)].copy()
        
        # Get the filtered unique regions
        unique_regions = merged_df[cfg.rcode_col].unique()
        unique_regions = [r for r in unique_regions if pd.notna(r)]
        
        # Check if any of the specified rcodes were found
        found_rcodes = set(unique_regions)
        requested_rcodes = set(rcodes_list)
        missing_rcodes = requested_rcodes - found_rcodes
        
        if missing_rcodes:
            print(f"⚠️  Warning: The following rcodes were not found in the data: {missing_rcodes}")
        
        if len(unique_regions) == 0:
            raise ValueError("❌ None of the specified rcodes were found in the data")
        
        print(f"✓ Found {len(unique_regions)} rcodes in the data: {unique_regions}")
        print(f"✓ Filtered data shape: {merged_df.shape}")
        
        # Fill missing data by region
        merged_df = fillMissingDataByRegion(merged_df, cfg)
        
        # Create forward-looking variables
        merged_df = createForwardLookingVariables(merged_df, cfg)
        
        # Create train/test tags
        merged_df = createTrainTestTags(merged_df, cfg)
        
        # Add engineered features
        merged_df = addAllFeatures(merged_df, cfg)
        
        print(f"\nProcessing {len(unique_regions)} MSA regions for test...")
        
        # Process each MSA region
        processed_count = 0
        skipped_count = 0
        
        for i, region in enumerate(unique_regions, 1):
            print(f"\n[{i}/{len(unique_regions)}] Processing MSA region: {region}")
            
            # Filter data for this region
            region_df = merged_df[merged_df[cfg.rcode_col] == region].copy()
            
            # Process the region
            region_results = processRegionMSA(region, region_df, cfg, target_col='HPA1Yfwd')
            
            # Store results
            REGION_RESULTS[region] = region_results
            
            if region_results is not None:
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"\n{'='*60}")
        print("MSA TEST PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {processed_count} MSA regions")
        print(f"⚠️  Skipped: {skipped_count} MSA regions")
        print(f"📊 Total regions processed: {len(unique_regions)}")
        
        # Generate final output
        if processed_count > 0:
            final_output = generateFinalOutput(merged_df, cfg)
            
            # Save final output with error handling
            try:
                print(f"Saving test output to: {cfg.output_path}")
                final_output.to_csv(cfg.output_path, index=False)
                
                # Verify the file was saved successfully
                if os.path.exists(cfg.output_path):
                    file_size = os.path.getsize(cfg.output_path)
                    logger.info(f"Test output saved successfully to: {cfg.output_path} ({file_size} bytes)")
                    print(f"✓ Test output saved to: {cfg.output_path} ({file_size / 1024:.1f} KB)")
                else:
                    raise FileNotFoundError("Output file was not created successfully")
                    
            except PermissionError:
                error_msg = f"❌ Permission denied: Cannot write to {cfg.output_path}"
                logger.error(error_msg)
                print(error_msg)
                print("  Try running with administrator privileges or choose a different output location.")
                raise
            except OSError as e:
                error_msg = f"❌ OS Error writing file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            except Exception as e:
                error_msg = f"❌ Unexpected error saving file: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                raise
            
            # Display sample of final output
            print(f"\n{'='*50}")
            print("SAMPLE OF TEST OUTPUT")
            print(f"{'='*50}")
            print(final_output.head(10))
            
            print(f"\nTest output columns: {list(final_output.columns)}")
            print(f"Test output shape: {final_output.shape}")
            print(f"Date range: {final_output['Year_Month_Day'].min()} to {final_output['Year_Month_Day'].max()}")
            print(f"Test MSAs: {final_output['rcode'].nunique()}")
            print(f"Train records: {(final_output['tag'] == 'Train').sum()}")
            print(f"Test records: {(final_output['tag'] == 'Test').sum()}")
            
        else:
            print("❌ No MSA regions were successfully processed")
        
        print(f"\n{'='*70}")
        print("MSA BASELINE TEST PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
        return final_output
        
    except Exception as e:
        logger.error(f"Test pipeline execution failed: {str(e)}")
        print(f"❌ Test pipeline failed: {str(e)}")
        raise e

# Example usage of the test function
def example_test_usage():
    """
    Example of how to use the run_test_with_rcodes function for testing.
    """
    # Define a list of rcodes to test
    test_rcodes = ['MSA_12345', 'MSA_23456', 'MSA_34567']
    
    # Example model parameters for testing
    test_model_params = {
        'models': ['Ridge', 'RandomForest', 'XGBoost'],
        'model_params': {
            'Ridge': {'alpha': [1.0]},
            'RandomForest': {'n_estimators': [100], 'max_depth': [10]},
            'XGBoost': {'n_estimators': [100], 'max_depth': [6], 'learning_rate': [0.1]}
        },
        'lag_list': [1, 3, 6, 12],
        'rate_list': [1, 3, 6],
        'moving_averages': [3, 6, 12],
        'target_forward': 12
    }
    
    print("Example test run with specific rcodes:")
    print(f"Test rcodes: {test_rcodes}")
    
    # Run the test
    result = run_test_with_rcodes(
        rcodes_list=test_rcodes,
        msa_file='your_msa_data.csv',
        usa_file='your_usa_data.csv',
        output_file='testBaseline_Results.csv',
        end_date='2024-12-31',
        model_params=test_model_params
    )
    
    if result is not None:
        print("✓ Test run completed successfully!")
        print(f"✓ Output file: testBaseline_Results.csv")
        print(f"✓ Output shape: {result.shape}")
    else:
        print("❌ Test run failed")

# Uncomment the line below to run the example
# example_test_usage()