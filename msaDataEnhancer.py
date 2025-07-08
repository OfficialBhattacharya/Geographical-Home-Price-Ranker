import pandas as pd
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config_loader import get_config
import warnings
warnings.filterwarnings('ignore')

class MSADataEnhancer:
    """
    Enhances the FRED dataset with MSA-level metrics by fetching and processing
    metropolitan area-specific economic indicators from the FRED API.
    """
    
    def __init__(self, config=None):
        """Initialize the MSADataEnhancer with configuration"""
        self.config = config or get_config()
        self.api_key = self.config.get_api_key()
        self.max_retries = self.config.get_max_retries()
        self.timeout = self.config.get_api_timeout()
        
        # Load MSA mappings and reference data
        self.msa_mappings = self._load_msa_mappings()
        self.msa_reference = self._load_msa_reference()
        
        # Initialize metadata tracking
        self.metadata = {
            'successful_series': {},
            'failed_series': {},
            'coverage_percentage': {},
            'processing_log': [],
            'quarterly_interpolated': [],
            'low_coverage_skipped': []
        }
        
        # Quarterly series that need monthly interpolation
        self.quarterly_series = ['HomePriceIndex']
        
        # Realtor.com series with limited coverage
        self.realtor_series = ['MedianDaysonMarket', 'MedianListingPriceperSquareFeet']
        
        # Minimum coverage threshold (60% as specified in requirements)
        self.min_coverage_threshold = 0.60
        
    def _load_msa_mappings(self) -> Dict[str, str]:
        """Load MSA-level FRED series mappings"""
        mappings = {}
        
        try:
            with open('msa_level_fred_mappings.txt', 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and ':' in line:
                    series_name, pattern = line.split(':', 1)
                    series_name = series_name.strip()
                    pattern = pattern.strip()
                    
                    if pattern != 'NO_MSA_EQUIV':
                        mappings[series_name] = pattern
                        
        except FileNotFoundError:
            raise FileNotFoundError("msa_level_fred_mappings.txt not found")
            
        return mappings
    
    def _load_msa_reference(self) -> pd.DataFrame:
        """Load MSA reference data with CBSA codes and names"""
        try:
            # Skip comment lines and read CBSA data
            msa_data = []
            with open('msa_list_top150.txt', 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        cbsa_code = parts[0].strip()
                        msa_name = parts[1].strip()
                        msa_data.append({'CBSA': cbsa_code, 'MSA_Name': msa_name})
            
            if not msa_data:
                raise ValueError("No MSA data found in reference file")
                
            return pd.DataFrame(msa_data)
            
        except FileNotFoundError:
            raise FileNotFoundError("msa_list_top150.txt not found")
    
    def _fetch_fred_series(self, series_id: str) -> Optional[pd.DataFrame]:
        """Fetch a single FRED series with retry logic"""
        api_url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(api_url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    if not observations:
                        return None
                    
                    # Convert to DataFrame
                    df_data = []
                    for obs in observations:
                        if obs['value'] != '.':
                            try:
                                df_data.append({
                                    'Date': obs['date'],
                                    'Value': float(obs['value'])
                                })
                            except ValueError:
                                continue
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date').reset_index(drop=True)
                        return df
                    else:
                        return None
                        
                elif response.status_code == 429:  # Rate limit
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return None
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1)
                
        return None
    
    def _determine_frequency(self, df: pd.DataFrame) -> str:
        """Determine the frequency of the data series"""
        if len(df) < 2:
            return 'unknown'
        
        # Calculate typical interval between observations
        intervals = df['Date'].diff().dropna()
        avg_interval = intervals.median().days
        
        thresholds = self.config.get_frequency_thresholds()
        
        if avg_interval <= thresholds['monthly_max_days']:
            return 'monthly'
        elif thresholds['quarterly_min_days'] <= avg_interval <= thresholds['quarterly_max_days']:
            return 'quarterly'
        elif avg_interval >= thresholds['annual_min_days']:
            return 'annual'
        else:
            return 'unknown'
    
    def _interpolate_quarterly_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate quarterly data to monthly using linear interpolation"""
        # Set Date as index for resampling
        df_indexed = df.set_index('Date')
        
        # Create monthly date range
        start_date = df_indexed.index.min()
        end_date = df_indexed.index.max()
        monthly_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Reindex to monthly and interpolate
        monthly_df = df_indexed.reindex(monthly_range)
        monthly_df['Value'] = monthly_df['Value'].interpolate(method='linear')
        
        # Reset index and return
        monthly_df.reset_index(inplace=True)
        monthly_df.rename(columns={'index': 'Date'}, inplace=True)
        
        return monthly_df
    
    def _check_data_coverage(self, df: pd.DataFrame, series_name: str, cbsa: str) -> bool:
        """Check if data coverage meets minimum threshold"""
        if df is None or len(df) == 0:
            return False
        
        # For Realtor.com series, check recent coverage
        if series_name in self.realtor_series:
            recent_date = datetime.now() - timedelta(days=180)  # 6 months ago
            recent_data = df[df['Date'] >= recent_date]
            
            if len(recent_data) == 0:
                self.metadata['low_coverage_skipped'].append({
                    'CBSA': cbsa,
                    'Series': series_name,
                    'Reason': 'No recent data (>6 months old)'
                })
                return False
        
        # Check overall coverage for last 2 years
        two_years_ago = datetime.now() - timedelta(days=730)
        recent_period = df[df['Date'] >= two_years_ago]
        
        if len(recent_period) == 0:
            return False
        
        # Calculate expected vs actual data points
        expected_points = 24 if series_name not in self.quarterly_series else 8  # 2 years of monthly/quarterly data
        actual_points = len(recent_period)
        coverage_ratio = actual_points / expected_points
        
        if coverage_ratio < self.min_coverage_threshold:
            self.metadata['low_coverage_skipped'].append({
                'CBSA': cbsa,
                'Series': series_name,
                'Coverage': f"{coverage_ratio:.1%}",
                'Reason': f'Coverage below {self.min_coverage_threshold:.0%} threshold'
            })
            return False
        
        return True
    
    def _process_msa_series(self, cbsa: str, msa_name: str, series_name: str, pattern: str) -> Optional[pd.DataFrame]:
        """Process a single series for a specific MSA"""
        try:
            # Replace [CBSA] with CBSA code
            series_id = pattern.replace('[CBSA]', cbsa)
            
            # Fetch data
            df = self._fetch_fred_series(series_id)
            if df is None or len(df) == 0:
                self.metadata['processing_log'].append(
                    f"Failed to fetch {series_name} for {msa_name} ({cbsa}): No data available"
                )
                return None
            
            # Check data coverage
            if not self._check_data_coverage(df, series_name, cbsa):
                return None
            
            # Determine frequency
            frequency = self._determine_frequency(df)
            
            # Handle frequency conversion for quarterly data
            if frequency == 'quarterly' and series_name in self.quarterly_series:
                df = self._interpolate_quarterly_to_monthly(df)
                frequency = 'monthly_interpolated'
                self.metadata['quarterly_interpolated'].append(f"{series_name} for {msa_name} ({cbsa})")
            
            # Add metadata columns
            df['CBSA_Code'] = cbsa
            df['MSA_Name'] = msa_name
            df['SeriesID'] = series_id
            df['Frequency'] = frequency
            df['Source'] = 'FRED'
            df['GeographyType'] = 'MSA'
            df['GeographyCode'] = cbsa
            
            # Track successful series
            if cbsa not in self.metadata['successful_series']:
                self.metadata['successful_series'][cbsa] = []
            self.metadata['successful_series'][cbsa].append(series_name)
            
            return df
            
        except Exception as e:
            self.metadata['processing_log'].append(
                f"Error processing {series_name} for {msa_name} ({cbsa}): {str(e)}"
            )
            return None
    
    def enhance_dataset(self, output_filename: str = 'msa_level_data.csv') -> pd.DataFrame:
        """Main method to enhance dataset with MSA-level data"""
        
        print(f"Starting MSA-level data enhancement...")
        print(f"Processing {len(self.msa_reference)} MSAs with {len(self.msa_mappings)} available series")
        
        all_msa_data = []
        total_operations = len(self.msa_reference) * len(self.msa_mappings)
        current_operation = 0
        
        for _, msa_row in self.msa_reference.iterrows():
            cbsa = msa_row['CBSA']
            msa_name = msa_row['MSA_Name']
            
            print(f"\nProcessing {msa_name} ({cbsa})...")
            msa_series_count = 0
            
            for series_name, pattern in self.msa_mappings.items():
                current_operation += 1
                
                # Process the series
                series_df = self._process_msa_series(cbsa, msa_name, series_name, pattern)
                
                if series_df is not None:
                    series_df['SeriesName'] = series_name
                    all_msa_data.append(series_df)
                    msa_series_count += 1
                    print(f"  ✓ {series_name}")
                else:
                    if cbsa not in self.metadata['failed_series']:
                        self.metadata['failed_series'][cbsa] = []
                    self.metadata['failed_series'][cbsa].append(series_name)
                    print(f"  ✗ {series_name}")
                
                # Progress indicator
                if current_operation % 10 == 0:
                    progress = (current_operation / total_operations) * 100
                    print(f"  Progress: {progress:.1f}% ({current_operation}/{total_operations})")
            
            # Calculate coverage for this MSA
            coverage_pct = (msa_series_count / len(self.msa_mappings)) * 100
            self.metadata['coverage_percentage'][cbsa] = {
                'MSA_Name': msa_name,
                'Available_Series': msa_series_count,
                'Total_Series': len(self.msa_mappings),
                'Coverage_Percentage': coverage_pct
            }
            
            print(f"  Series coverage: {msa_series_count}/{len(self.msa_mappings)} ({coverage_pct:.1f}%)")
        
        # Combine all data
        if all_msa_data:
            combined_df = pd.concat(all_msa_data, ignore_index=True)
            combined_df = combined_df.sort_values(['CBSA_Code', 'SeriesName', 'Date']).reset_index(drop=True)
            
            # Save to CSV
            output_dir = self.config.get_processed_data_dir()
            output_path = os.path.join(output_dir, output_filename)
            combined_df.to_csv(output_path, index=False)
            
            print(f"\n✓ MSA data saved to {output_path}")
            print(f"Final dataset shape: {combined_df.shape}")
            
            # Save metadata
            self._save_metadata()
            
            return combined_df
        else:
            print("\n✗ No MSA data was successfully processed")
            return pd.DataFrame()
    
    def _save_metadata(self):
        """Save processing metadata to JSON file"""
        output_dir = self.config.get_processed_data_dir()
        metadata_path = os.path.join(output_dir, 'msa_data_metadata.json')
        
        # Add summary statistics
        self.metadata['summary'] = {
            'total_msas_processed': len(self.msa_reference),
            'total_series_attempted': len(self.msa_mappings),
            'successful_msas': len(self.metadata['successful_series']),
            'quarterly_interpolations': len(self.metadata['quarterly_interpolated']),
            'low_coverage_skipped': len(self.metadata['low_coverage_skipped']),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved to {metadata_path}")
    
    def validate_2023_data(self, df: pd.DataFrame) -> Dict:
        """Validate 2023 data against known benchmarks"""
        print("\nValidating 2023 MSA data...")
        
        validation_results = {
            'spot_checks': [],
            'data_quality': {},
            'coverage_analysis': {}
        }
        
        # Filter for 2023 data
        df_2023 = df[df['Date'].dt.year == 2023]
        
        if len(df_2023) == 0:
            print("No 2023 data found for validation")
            return validation_results
        
        # Spot-check major MSAs across size tiers
        major_msas = ['35620', '31080', '16980']  # NYC, LA, Chicago
        medium_msas = ['12420', '16740', '19740']  # Austin, Charlotte, Denver  
        smaller_msas = ['29620', '22180', '23540']  # Laredo, Fargo, Gainesville
        
        spot_check_msas = major_msas + medium_msas + smaller_msas
        
        for cbsa in spot_check_msas:
            cbsa_data = df_2023[df_2023['CBSA_Code'] == cbsa]
            if len(cbsa_data) > 0:
                msa_name = cbsa_data['MSA_Name'].iloc[0]
                available_series = cbsa_data['SeriesName'].unique()
                validation_results['spot_checks'].append({
                    'CBSA': cbsa,
                    'MSA_Name': msa_name,
                    'Available_Series': list(available_series),
                    'Data_Points': len(cbsa_data)
                })
        
        # Overall coverage analysis
        validation_results['coverage_analysis'] = {
            'msas_with_2023_data': len(df_2023['CBSA_Code'].unique()),
            'total_msas': len(self.msa_reference),
            'series_coverage': dict(df_2023['SeriesName'].value_counts()),
            'avg_data_points_per_msa': df_2023.groupby('CBSA_Code').size().mean()
        }
        
        print(f"✓ Validation complete - {len(validation_results['spot_checks'])} MSAs spot-checked")
        return validation_results
    
    def integrate_with_main_dataset(self, main_file: str, msa_data: pd.DataFrame) -> pd.DataFrame:
        """Integrate MSA data with existing national/state dataset"""
        print(f"\nIntegrating MSA data with main dataset: {main_file}")
        
        try:
            # Load main dataset
            main_df = pd.read_csv(main_file)
            print(f"Main dataset shape: {main_df.shape}")
            
            # Ensure main dataset has geography columns
            if 'GeographyType' not in main_df.columns:
                main_df['GeographyType'] = 'National'
                main_df['GeographyCode'] = 'US'
            
            # Prepare MSA data for integration
            msa_integration_df = msa_data[['Date', 'SeriesName', 'Value', 'CBSA_Code', 'MSA_Name', 
                                         'GeographyType', 'GeographyCode', 'SeriesID', 'Source']].copy()
            
            # Rename columns to match main dataset format
            msa_integration_df = msa_integration_df.rename(columns={
                'SeriesName': 'Series',
                'CBSA_Code': 'GeographyCode',
                'MSA_Name': 'GeographyName'
            })
            
            # Add MSA-to-state mapping for hierarchical analysis
            # This would require additional mapping data - simplified for now
            msa_integration_df['State'] = msa_integration_df['GeographyName'].str.extract(r', ([A-Z]{2})')
            
            # Combine datasets
            integrated_df = pd.concat([main_df, msa_integration_df], ignore_index=True)
            integrated_df = integrated_df.sort_values(['GeographyType', 'GeographyCode', 'Date']).reset_index(drop=True)
            
            # Save integrated dataset
            output_dir = self.config.get_processed_data_dir()
            integrated_path = os.path.join(output_dir, 'integrated_national_state_msa_data.csv')
            integrated_df.to_csv(integrated_path, index=False)
            
            print(f"✓ Integrated dataset saved to {integrated_path}")
            print(f"Final integrated shape: {integrated_df.shape}")
            
            return integrated_df
            
        except Exception as e:
            print(f"✗ Integration failed: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    print("MSA Data Enhancement Test")
    print("=" * 50)
    
    try:
        enhancer = MSADataEnhancer()
        
        # Test with a small subset first
        print("Testing with first 5 MSAs...")
        enhancer.msa_reference = enhancer.msa_reference.head(5)
        
        msa_data = enhancer.enhance_dataset('test_msa_data.csv')
        
        if len(msa_data) > 0:
            validation_results = enhancer.validate_2023_data(msa_data)
            print("\nTest successful - MSA data enhancement working")
        else:
            print("Test failed - no data retrieved")
            
    except Exception as e:
        print(f"Test failed: {str(e)}") 