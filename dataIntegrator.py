import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config_loader import get_config
import warnings
warnings.filterwarnings('ignore')

class GeographicalDataIntegrator:
    """
    Comprehensive data integration utility for combining National, State, and MSA-level
    economic data into a unified hierarchical dataset with GeographyType dimension.
    """
    
    def __init__(self, config=None):
        """Initialize the data integrator with configuration"""
        self.config = config or get_config()
        
        # Initialize metadata tracking
        self.integration_metadata = {
            'source_files': {},
            'processing_log': [],
            'geography_mapping': {},
            'data_quality': {},
            'final_summary': {}
        }
        
        # Geography hierarchy mapping
        self.geography_hierarchy = {
            'National': {'code': 'US', 'parent': None},
            'State': {'parent': 'National'},
            'MSA': {'parent': 'State'}
        }
    
    def _load_msa_to_state_mapping(self) -> Dict[str, str]:
        """Create MSA-to-state mapping from MSA names"""
        mapping = {}
        
        try:
            if os.path.exists('msa_list_top150.txt'):
                with open('msa_list_top150.txt', 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            cbsa_code = parts[0].strip()
                            msa_name = parts[1].strip()
                            
                            # Extract state codes from MSA name (e.g., "NY-NJ-PA" -> ["NY", "NJ", "PA"])
                            # Use the first state as primary
                            if ', ' in msa_name:
                                state_part = msa_name.split(', ')[-1]
                                # Handle cases like "NY-NJ-PA" or "CA"
                                if '-' in state_part:
                                    primary_state = state_part.split('-')[0]
                                else:
                                    primary_state = state_part
                                
                                if len(primary_state) == 2:  # Valid state abbreviation
                                    mapping[cbsa_code] = primary_state
            
            self.integration_metadata['geography_mapping']['msa_to_state'] = mapping
            return mapping
            
        except Exception as e:
            self.integration_metadata['processing_log'].append(f"Error creating MSA-to-state mapping: {str(e)}")
            return {}
    
    def _standardize_dataset_structure(self, df: pd.DataFrame, geography_type: str) -> pd.DataFrame:
        """Standardize dataset structure for integration"""
        
        # Ensure required columns exist
        required_columns = ['Date', 'Value']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {geography_type} dataset")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure Date is first of month for consistency
        df['Date'] = df['Date'].dt.to_period('M').dt.start_time
        
        # Add/standardize geography columns
        if 'GeographyType' not in df.columns:
            df['GeographyType'] = geography_type
        
        # Handle geography codes and names based on type
        if geography_type == 'National':
            df['GeographyCode'] = 'US'
            df['GeographyName'] = 'United States'
            df['State'] = None
            df['ParentGeography'] = None
            
        elif geography_type == 'State':
            # Ensure State column exists
            if 'State' not in df.columns:
                # Try to extract from existing columns
                if 'GeographyCode' in df.columns:
                    df['State'] = df['GeographyCode']
                else:
                    raise ValueError("State geography data missing State identifier")
            
            df['GeographyCode'] = df['State']
            df['GeographyName'] = df['State']  # Can be enhanced with full state names
            df['ParentGeography'] = 'US'
            
        elif geography_type == 'MSA':
            # Handle MSA-specific columns
            if 'CBSA_Code' in df.columns:
                df['GeographyCode'] = df['CBSA_Code']
            elif 'GeographyCode' not in df.columns:
                raise ValueError("MSA geography data missing CBSA code")
            
            if 'MSA_Name' in df.columns:
                df['GeographyName'] = df['MSA_Name']
            elif 'GeographyName' not in df.columns:
                df['GeographyName'] = df['GeographyCode']  # Fallback
            
            # Add MSA-to-state mapping
            msa_to_state = self._load_msa_to_state_mapping()
            df['State'] = df['GeographyCode'].map(msa_to_state)
            df['ParentGeography'] = df['State']
        
        # Ensure SeriesName exists
        if 'SeriesName' not in df.columns:
            # Try to infer from other columns
            series_cols = [col for col in df.columns if col not in 
                          ['Date', 'Value', 'GeographyType', 'GeographyCode', 'GeographyName', 
                           'State', 'ParentGeography', 'Source', 'SeriesID', 'Frequency']]
            
            if len(series_cols) == 1:
                # Melt the dataset if it has one data column
                df = df.melt(
                    id_vars=[col for col in df.columns if col != series_cols[0]],
                    value_vars=series_cols,
                    var_name='SeriesName',
                    value_name='Value_New'
                )
                df['Value'] = df['Value_New']
                df = df.drop('Value_New', axis=1)
            else:
                df['SeriesName'] = 'Unknown'
        
        # Add metadata columns if missing
        if 'Source' not in df.columns:
            df['Source'] = 'FRED'
        
        if 'LastUpdated' not in df.columns:
            df['LastUpdated'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def load_national_data(self, filename: str = None) -> Optional[pd.DataFrame]:
        """Load and standardize national-level data"""
        
        if filename is None:
            # Look for common national data files
            candidate_files = [
                'unified_monthly_data.csv',
                'national_data.csv',
                'processed_fred_data.csv'
            ]
            
            for candidate in candidate_files:
                if os.path.exists(candidate):
                    filename = candidate
                    break
        
        if filename is None or not os.path.exists(filename):
            self.integration_metadata['processing_log'].append("No national data file found")
            return None
        
        try:
            print(f"Loading national data from {filename}...")
            df = pd.read_csv(filename)
            
            # Check if this is already processed data with Region column
            if 'Region' in df.columns and df['Region'].iloc[0] == 'United States':
                # This appears to be national-level data - melt it
                id_cols = ['Date', 'Region']
                value_cols = [col for col in df.columns if col not in id_cols]
                
                df_melted = df.melt(
                    id_vars=id_cols,
                    value_vars=value_cols,
                    var_name='SeriesName',
                    value_name='Value'
                )
                df_melted = df_melted.dropna(subset=['Value'])
                df = df_melted
            
            standardized_df = self._standardize_dataset_structure(df, 'National')
            
            self.integration_metadata['source_files']['national'] = {
                'filename': filename,
                'records': len(standardized_df),
                'date_range': f"{standardized_df['Date'].min()} to {standardized_df['Date'].max()}",
                'series_count': standardized_df['SeriesName'].nunique()
            }
            
            print(f"✓ Loaded {len(standardized_df):,} national records")
            return standardized_df
            
        except Exception as e:
            self.integration_metadata['processing_log'].append(f"Error loading national data: {str(e)}")
            print(f"✗ Error loading national data: {e}")
            return None
    
    def load_state_data(self, filename: str = 'state_level_data.csv') -> Optional[pd.DataFrame]:
        """Load and standardize state-level data"""
        
        if not os.path.exists(filename):
            self.integration_metadata['processing_log'].append(f"State data file not found: {filename}")
            return None
        
        try:
            print(f"Loading state data from {filename}...")
            df = pd.read_csv(filename)
            
            standardized_df = self._standardize_dataset_structure(df, 'State')
            
            self.integration_metadata['source_files']['state'] = {
                'filename': filename,
                'records': len(standardized_df),
                'states': standardized_df['State'].nunique(),
                'date_range': f"{standardized_df['Date'].min()} to {standardized_df['Date'].max()}",
                'series_count': standardized_df['SeriesName'].nunique()
            }
            
            print(f"✓ Loaded {len(standardized_df):,} state records for {standardized_df['State'].nunique()} states")
            return standardized_df
            
        except Exception as e:
            self.integration_metadata['processing_log'].append(f"Error loading state data: {str(e)}")
            print(f"✗ Error loading state data: {e}")
            return None
    
    def load_msa_data(self, filename: str = 'msa_level_data.csv') -> Optional[pd.DataFrame]:
        """Load and standardize MSA-level data"""
        
        if not os.path.exists(filename):
            self.integration_metadata['processing_log'].append(f"MSA data file not found: {filename}")
            return None
        
        try:
            print(f"Loading MSA data from {filename}...")
            df = pd.read_csv(filename)
            
            standardized_df = self._standardize_dataset_structure(df, 'MSA')
            
            self.integration_metadata['source_files']['msa'] = {
                'filename': filename,
                'records': len(standardized_df),
                'msas': standardized_df['GeographyCode'].nunique(),
                'date_range': f"{standardized_df['Date'].min()} to {standardized_df['Date'].max()}",
                'series_count': standardized_df['SeriesName'].nunique()
            }
            
            print(f"✓ Loaded {len(standardized_df):,} MSA records for {standardized_df['GeographyCode'].nunique()} MSAs")
            return standardized_df
            
        except Exception as e:
            self.integration_metadata['processing_log'].append(f"Error loading MSA data: {str(e)}")
            print(f"✗ Error loading MSA data: {e}")
            return None
    
    def integrate_all_data(self, output_filename: str = 'integrated_national_state_msa_data.csv') -> pd.DataFrame:
        """Integrate all geographic levels into unified dataset"""
        
        print("\n=== GEOGRAPHICAL DATA INTEGRATION ===")
        print("Loading data from all geographic levels...")
        
        all_datasets = []
        
        # Load national data
        national_df = self.load_national_data()
        if national_df is not None:
            all_datasets.append(national_df)
        
        # Load state data
        state_df = self.load_state_data()
        if state_df is not None:
            all_datasets.append(state_df)
        
        # Load MSA data
        msa_df = self.load_msa_data()
        if msa_df is not None:
            all_datasets.append(msa_df)
        
        if not all_datasets:
            print("✗ No datasets found for integration")
            return pd.DataFrame()
        
        print(f"\nIntegrating {len(all_datasets)} dataset(s)...")
        
        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Ensure consistent column order
        column_order = [
            'Date', 'GeographyType', 'GeographyCode', 'GeographyName', 
            'State', 'ParentGeography', 'SeriesName', 'Value',
            'Source', 'SeriesID', 'Frequency', 'LastUpdated'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in column_order if col in combined_df.columns]
        other_columns = [col for col in combined_df.columns if col not in available_columns]
        final_columns = available_columns + other_columns
        
        combined_df = combined_df[final_columns]
        
        # Sort by geography hierarchy, then by date
        geography_order = {'National': 1, 'State': 2, 'MSA': 3}
        combined_df['_sort_order'] = combined_df['GeographyType'].map(geography_order)
        combined_df = combined_df.sort_values(['_sort_order', 'GeographyCode', 'SeriesName', 'Date'])
        combined_df = combined_df.drop('_sort_order', axis=1).reset_index(drop=True)
        
        # Save integrated dataset
        output_dir = self.config.get_processed_data_dir()
        output_path = os.path.join(output_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        
        # Generate summary statistics
        self._generate_integration_summary(combined_df)
        
        print(f"\n✓ Integration complete!")
        print(f"Final dataset saved to: {output_path}")
        print(f"Total records: {len(combined_df):,}")
        
        return combined_df
    
    def _generate_integration_summary(self, df: pd.DataFrame):
        """Generate comprehensive integration summary"""
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'geography_breakdown': dict(df['GeographyType'].value_counts()),
            'series_coverage': dict(df['SeriesName'].value_counts()),
            'geographic_entities': {
                'national': len(df[df['GeographyType'] == 'National']['GeographyCode'].unique()),
                'states': len(df[df['GeographyType'] == 'State']['GeographyCode'].unique()),
                'msas': len(df[df['GeographyType'] == 'MSA']['GeographyCode'].unique())
            }
        }
        
        # Data quality checks
        quality_checks = {
            'missing_values': int(df['Value'].isnull().sum()),
            'duplicate_records': int(df.duplicated().sum()),
            'date_consistency': df['Date'].dt.day.nunique() == 1,  # Should all be first of month
            'geographic_hierarchy_completeness': self._check_hierarchy_completeness(df)
        }
        
        self.integration_metadata['final_summary'] = summary
        self.integration_metadata['data_quality'] = quality_checks
        
        # Save metadata
        metadata_path = os.path.join(self.config.get_processed_data_dir(), 'integration_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.integration_metadata, f, indent=2, default=str)
        
        # Print summary
        print(f"\nIntegration Summary:")
        print(f"  Total Records: {summary['total_records']:,}")
        print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Geography Breakdown:")
        for geo_type, count in summary['geography_breakdown'].items():
            print(f"    - {geo_type}: {count:,} records")
        print(f"  Geographic Entities:")
        print(f"    - National: {summary['geographic_entities']['national']}")
        print(f"    - States: {summary['geographic_entities']['states']}")
        print(f"    - MSAs: {summary['geographic_entities']['msas']}")
        
        print(f"\nData Quality:")
        print(f"  Missing Values: {quality_checks['missing_values']:,}")
        print(f"  Duplicate Records: {quality_checks['duplicate_records']}")
        print(f"  Date Consistency: {'✓' if quality_checks['date_consistency'] else '✗'}")
        
        print(f"\nMetadata saved to: {metadata_path}")
    
    def _check_hierarchy_completeness(self, df: pd.DataFrame) -> Dict:
        """Check completeness of geographic hierarchy"""
        
        # Check MSA-to-state mapping completeness
        msa_df = df[df['GeographyType'] == 'MSA']
        msas_with_state = msa_df['State'].notna().sum()
        total_msas = len(msa_df['GeographyCode'].unique())
        
        return {
            'msa_state_mapping_completeness': f"{msas_with_state}/{len(msa_df)} MSA records have state mapping",
            'unique_msa_state_completeness': f"{msa_df['State'].nunique()}/{total_msas} unique MSAs mapped to states"
        }
    
    def create_hierarchical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical summary for analysis purposes"""
        
        # Summary by geography type and series
        summary_df = df.groupby(['GeographyType', 'SeriesName', 'Date']).agg({
            'Value': ['count', 'mean', 'std', 'min', 'max'],
            'GeographyCode': 'nunique'
        }).round(2)
        
        summary_df.columns = ['Record_Count', 'Mean_Value', 'Std_Value', 'Min_Value', 'Max_Value', 'Geographic_Entities']
        summary_df = summary_df.reset_index()
        
        return summary_df

def run_complete_integration():
    """Run complete data integration process"""
    
    print("Geographical Data Integration Pipeline")
    print("=" * 60)
    
    try:
        # Initialize integrator
        integrator = GeographicalDataIntegrator()
        
        # Run integration
        integrated_data = integrator.integrate_all_data()
        
        if len(integrated_data) > 0:
            print("\n✓ Integration successful!")
            
            # Create hierarchical summary
            summary_df = integrator.create_hierarchical_summary(integrated_data)
            summary_path = os.path.join(integrator.config.get_processed_data_dir(), 'hierarchical_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Hierarchical summary saved to: {summary_path}")
            
            return True
        else:
            print("\n✗ Integration failed - no data processed")
            return False
            
    except Exception as e:
        print(f"\n✗ Integration failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Geographical Data Integration Utility")
    print("=" * 50)
    
    success = run_complete_integration()
    
    if success:
        print("\n✓ Data integration completed successfully!")
    else:
        print("\n✗ Data integration failed!")
        exit(1) 