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

class CFG:
    """
    Configuration class for MSA Baseline model with USA data integration.
    """
    
    def __init__(self, get_user_input=True):
        logger.info("Initializing CFG class for MSA Baseline with USA data...")
        
        if get_user_input:
            self._get_user_configurations()
        else:
            self._set_default_configurations()
        
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
        self.end_date = input("Enter end date (YYYY-MM-DD) [2025-01-01]: ") or "2025-01-01"
        
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
        self.end_date = "2025-01-01"
        
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

def loadAndMergeData(cfg):
    """
    Load MSA and USA data, then merge them.
    
    Parameters:
    - cfg: Configuration object
    
    Returns:
    - merged_df: Merged dataframe with both MSA and USA data
    - unique_regions: List of unique MSA regions
    """
    logger.info("Step 1: Loading and merging MSA and USA data...")
    print("Loading MSA and USA raw data...")
    
    try:
        # Load MSA data
        print(f"Loading MSA data from: {cfg.msa_file_path}")
        msa_df = pd.read_csv(cfg.msa_file_path)
        logger.info(f"MSA data loaded. Shape: {msa_df.shape}")
        print(f"MSA data shape: {msa_df.shape}")
        
        # Load USA data
        print(f"Loading USA data from: {cfg.usa_file_path}")
        usa_df = pd.read_csv(cfg.usa_file_path)
        logger.info(f"USA data loaded. Shape: {usa_df.shape}")
        print(f"USA data shape: {usa_df.shape}")
        
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

def createForwardLookingVariables(df, cfg):
    """
    Create forward-looking variables (HPA1Yfwd, HPI1Y_fwd) for each MSA.
    
    Parameters:
    - df: Input dataframe
    - cfg: Configuration object
    
    Returns:
    - df: Dataframe with forward-looking variables
    """
    logger.info("Step 2: Creating forward-looking variables...")
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
    logger.info("Step 3: Creating train/test tags...")
    print("Creating train/test split tags...")
    
    df_tagged = df.copy()
    
    # Initialize tag column
    df_tagged['tag'] = 'train'
    
    # For each MSA, mark the last 12 months (where HPA1Yfwd is NaN) as test
    for region in df_tagged[cfg.rcode_col].unique():
        if pd.isna(region):
            continue
            
        region_mask = df_tagged[cfg.rcode_col] == region
        region_df = df_tagged[region_mask].copy()
        
        # Find rows where HPA1Yfwd is NaN (these are test rows)
        test_mask = region_df['HPA1Yfwd'].isna()
        
        # Update the main dataframe
        df_tagged.loc[region_mask & df_tagged['HPA1Yfwd'].isna(), 'tag'] = 'test'
    
    train_count = (df_tagged['tag'] == 'train').sum()
    test_count = (df_tagged['tag'] == 'test').sum()
    
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
    logger.info("Step 4: Adding engineered features...")
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
        train_df = df_region[df_region['tag'] == 'train'].copy()
        test_df = df_region[df_region['tag'] == 'test'].copy()
        
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
    logger.info("Step 5: Generating final output...")
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
        cfg = CFG(get_user_input=False)  # Set to True to get user input
        
        # Step 2: Load and merge data
        merged_df, unique_regions = loadAndMergeData(cfg)
        
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
            
            # Save final output
            final_output.to_csv(cfg.output_path, index=False)
            logger.info(f"Final output saved to: {cfg.output_path}")
            print(f"✓ Final output saved to: {cfg.output_path}")
            
            # Display sample of final output
            print(f"\n{'='*50}")
            print("SAMPLE OF FINAL OUTPUT")
            print(f"{'='*50}")
            print(final_output.head(10))
            
            print(f"\nOutput columns: {list(final_output.columns)}")
            print(f"Output shape: {final_output.shape}")
            print(f"Date range: {final_output['Year_Month_Day'].min()} to {final_output['Year_Month_Day'].max()}")
            print(f"Unique MSAs: {final_output['rcode'].nunique()}")
            print(f"Train records: {(final_output['tag'] == 'train').sum()}")
            print(f"Test records: {(final_output['tag'] == 'test').sum()}")
            
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

# Example usage with specific file paths
def run_with_custom_paths(msa_file, usa_file, output_file):
    """
    Run the MSA baseline model with custom file paths.
    
    Parameters:
    - msa_file: Path to MSA raw data CSV
    - usa_file: Path to USA raw data CSV  
    - output_file: Path for output CSV
    """
    global REGION_RESULTS
    REGION_RESULTS = {}  # Reset results
    
    # Create configuration with custom paths
    cfg = CFG(get_user_input=False)
    cfg.msa_file_path = msa_file
    cfg.usa_file_path = usa_file
    cfg.output_path = output_file
    
    print(f"Running MSA Baseline with:")
    print(f"  MSA data: {msa_file}")
    print(f"  USA data: {usa_file}")
    print(f"  Output: {output_file}")
    
    # Run the main pipeline
    main()

# Uncomment and modify the line below to run with your specific file paths
# run_with_custom_paths("path/to/msa_data.csv", "path/to/usa_data.csv", "output_results.csv")

"""
# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

Example of how to use the MSA Baseline script:

1. Basic usage with default settings:
   ```python
   # Make sure your data files are in the correct format
   # MSA data should have columns: Year_Month_Day, rcode, cs_name, HPI, hpa12m
   # USA data should have columns: Year_Month_Day, ProjectedHPA1YFwd_USABaseline, USA_HPA1Yfwd, USA_HPI1Yfwd
   
   main()  # This will use default file paths
   ```

2. Usage with custom file paths:
   ```python
   run_with_custom_paths(
       msa_file="path/to/your/msa_data.csv",
       usa_file="path/to/your/usa_data.csv", 
       output_file="path/to/your/output.csv"
   )
   ```

3. The output will be a CSV file with the following columns:
   - Year_Month_Day: Date in YYYY-MM-01 format
   - rcode: MSA region code
   - cs_name: MSA region name
   - tag: 'train' or 'test' indicator
   - ProjectedHPA1YFwd_USABaseline: USA baseline projections
   - ProjectedHPA1YFwd_MSABaseline: MSA baseline projections (model output)
   - HPI: Actual HPI values
   - hpa12m: Actual 12-month HPA values  
   - HPA1Yfwd: 1-year forward HPA (target variable)
   - HPI1Y_fwd: 1-year forward HPI
   - USA_HPA1Yfwd: USA 1-year forward HPA
   - USA_HPI1Yfwd: USA 1-year forward HPI

Expected Input Data Format:

MSA Raw Data (msa_data.csv):
```
Year_Month_Day,rcode,cs_name,HPI,hpa12m,[additional_features...]
2020-01-01,12345,Metro Area 1,250.5,0.045,...
2020-02-01,12345,Metro Area 1,252.1,0.047,...
...
```

USA Raw Data (usa_data.csv):
```
Year_Month_Day,ProjectedHPA1YFwd_USABaseline,USA_HPA1Yfwd,USA_HPI1Yfwd
2020-01-01,0.055,0.052,265.2
2020-02-01,0.057,0.054,267.1
...
```

Key Features:
- Automatically handles train/test split (last 12 months are test)
- Creates comprehensive engineered features (lags, moving averages, etc.)
- Uses ensemble modeling approach with neural network meta-learner
- Integrates USA-level baseline projections as features
- Outputs standardized format for downstream analysis

Performance Notes:
- Processing time depends on number of MSAs and time periods
- Memory usage scales with dataset size
- Consider processing in batches for very large datasets
"""

def create_sample_data_demo():
    """
    Create sample data to demonstrate the MSA baseline script functionality.
    """
    print("Creating sample MSA and USA data for demonstration...")
    
    # Create sample dates
    dates = pd.date_range(start='2020-01-01', end='2023-12-01', freq='MS')
    
    # Sample MSA data
    msa_regions = ['MSA_12345', 'MSA_23456', 'MSA_34567']
    region_names = ['Metro Area 1', 'Metro Area 2', 'Metro Area 3']
    
    msa_data = []
    for region, name in zip(msa_regions, region_names):
        for date in dates:
            # Simulate realistic housing data
            base_hpi = 250 + np.random.uniform(-50, 50)
            seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * (date.month - 1) / 12)
            trend_factor = 1 + 0.03 * (date.year - 2020)
            
            hpi = base_hpi * seasonal_factor * trend_factor + np.random.normal(0, 5)
            hpa12m = np.random.uniform(0.02, 0.08) + 0.01 * np.sin(2 * np.pi * (date.month - 1) / 12)
            
            msa_data.append({
                'Year_Month_Day': date,
                'rcode': region,
                'cs_name': name,
                'HPI': hpi,
                'hpa12m': hpa12m,
                'unemployment_rate': np.random.uniform(3, 7),
                'median_income': np.random.uniform(50000, 80000)
            })
    
    msa_df = pd.DataFrame(msa_data)
    
    # Sample USA data
    usa_data = []
    for date in dates:
        usa_data.append({
            'Year_Month_Day': date,
            'ProjectedHPA1YFwd_USABaseline': np.random.uniform(0.03, 0.07),
            'USA_HPA1Yfwd': np.random.uniform(0.025, 0.065),
            'USA_HPI1Yfwd': 260 + np.random.uniform(-10, 10)
        })
    
    usa_df = pd.DataFrame(usa_data)
    
    # Save sample data
    msa_df.to_csv('sample_msa_data.csv', index=False)
    usa_df.to_csv('sample_usa_data.csv', index=False)
    
    print(f"✓ Sample MSA data saved: sample_msa_data.csv ({msa_df.shape})")
    print(f"✓ Sample USA data saved: sample_usa_data.csv ({usa_df.shape})")
    print("✓ You can now run: run_with_custom_paths('sample_msa_data.csv', 'sample_usa_data.csv', 'sample_output.csv')")
    
    return msa_df, usa_df

# Uncomment the lines below to create and test with sample data
# create_sample_data_demo()
# run_with_custom_paths('sample_msa_data.csv', 'sample_usa_data.csv', 'sample_output.csv')