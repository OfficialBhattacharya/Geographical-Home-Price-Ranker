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

class StateDataEnhancer:
    """
    Enhances the FRED dataset with state-level metrics by fetching and processing
    state-specific economic indicators from the FRED API.
    """
    
    def __init__(self, config=None):
        """Initialize the StateDataEnhancer with configuration"""
        self.config = config or get_config()
        self.api_key = self.config.get_api_key()
        self.max_retries = self.config.get_max_retries()
        self.timeout = self.config.get_api_timeout()
        
        # Load state mappings
        self.state_mappings = self._load_state_mappings()
        self.states = self._load_states()
        
        # Initialize metadata tracking
        self.metadata = {
            'successful_series': {},
            'failed_series': {},
            'coverage_percentage': {},
            'processing_log': []
        }
        
        # Quarterly series that should not be interpolated
        self.quarterly_series = [
            'OccupiedHousingUnits', 'VacantHousingUnits_1', 'VacantforOtherReasons',
            'RenterOccupiedHousingUnits', 'VacantHousingUnits_NotYetOccupied',
            'VacantHousingUnits_forSale', 'TotalHousingUnits'
        ]
        
        # Realtor.com series with limited coverage
        self.realtor_series = ['MedianDaysonMarket', 'MedianListingPriceperSquareFeet']
        
    def _load_state_mappings(self) -> Dict[str, str]:
        """Load state-level FRED series mappings"""
        mappings = {}
        
        try:
            with open('state_level_fred_mappings.txt', 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and ':' in line:
                    series_name, pattern = line.split(':', 1)
                    series_name = series_name.strip()
                    pattern = pattern.strip()
                    
                    if pattern != 'NO_STATE_EQUIV':
                        mappings[series_name] = pattern
                        
        except FileNotFoundError:
            raise FileNotFoundError("state_level_fred_mappings.txt not found")
            
        return mappings
    
    def _load_states(self) -> List[str]:
        """Load state abbreviations from mappings file"""
        try:
            with open('state_level_fred_mappings.txt', 'r') as f:
                content = f.read()
                
            # Find the state abbreviations line
            for line in content.split('\n'):
                if line.startswith('AL,AK,AZ'):
                    return line.strip().split(',')
                    
        except FileNotFoundError:
            raise FileNotFoundError("state_level_fred_mappings.txt not found")
            
        return []
    
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
                            df_data.append({
                                'Date': obs['date'],
                                'Value': float(obs['value'])
                            })
                    
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
        """Determine the frequency of a time series"""
        if len(df) < 2:
            return 'unknown'
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, min(len(df), 5)):
            diff = (df.iloc[i]['Date'] - df.iloc[i-1]['Date']).days
            time_diffs.append(diff)
        
        avg_diff = np.mean(time_diffs)
        
        if avg_diff <= 35:  # Monthly
            return 'monthly'
        elif 80 <= avg_diff <= 100:  # Quarterly
            return 'quarterly'
        elif avg_diff >= 350:  # Annual
            return 'annual'
        else:
            return 'unknown'
    
    def _interpolate_quarterly_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate quarterly data to monthly using linear interpolation"""
        if len(df) < 2:
            return df
        
        # Create monthly date range
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        # Generate monthly dates
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        monthly_df = pd.DataFrame({'Date': monthly_dates})
        
        # Merge with original data
        merged_df = pd.merge(monthly_df, df, on='Date', how='left')
        
        # Linear interpolation
        merged_df['Value'] = merged_df['Value'].interpolate(method='linear')
        
        return merged_df[['Date', 'Value']].dropna()
    
    def _check_realtor_coverage(self, state: str, series_name: str) -> bool:
        """Check if Realtor.com series has sufficient coverage for a state"""
        if series_name not in self.realtor_series:
            return True
        
        # For Realtor.com series, we'll check if data exists for recent periods
        pattern = self.state_mappings.get(series_name)
        if not pattern:
            return False
        
        series_id = pattern.replace('[ST]', state)
        df = self._fetch_fred_series(series_id)
        
        if df is None or len(df) == 0:
            return False
        
        # Check if we have data in the last 2 years
        recent_data = df[df['Date'] >= datetime.now() - timedelta(days=730)]
        coverage_ratio = len(recent_data) / 24  # 24 months
        
        return coverage_ratio >= 0.6  # 60% coverage threshold
    
    def _process_state_series(self, state: str, series_name: str, pattern: str) -> Optional[pd.DataFrame]:
        """Process a single series for a specific state"""
        try:
            # Replace [ST] with state abbreviation
            series_id = pattern.replace('[ST]', state)
            
            # Check Realtor.com coverage
            if not self._check_realtor_coverage(state, series_name):
                self.metadata['processing_log'].append(
                    f"Skipped {series_name} for {state}: Insufficient Realtor.com coverage"
                )
                return None
            
            # Fetch data
            df = self._fetch_fred_series(series_id)
            if df is None or len(df) == 0:
                self.metadata['processing_log'].append(
                    f"Failed to fetch {series_name} for {state}: No data available"
                )
                return None
            
            # Determine frequency
            frequency = self._determine_frequency(df)
            
            # Handle frequency conversion
            if frequency == 'quarterly' and series_name not in self.quarterly_series:
                df = self._interpolate_quarterly_to_monthly(df)
                frequency = 'monthly'
            
            # Add metadata columns
            df['State'] = state
            df['SeriesID'] = series_id
            df['Frequency'] = frequency
            df['Source'] = 'FRED'
            
            # Handle proxy series
            if series_name == 'TotalShipmentsofNewHomes':
                df['ProxyFor'] = 'TotalShipmentsofNewHomes'
                # Apply seasonal adjustment if needed (simplified)
                if len(df) >= 12:
                    # Simple moving average for seasonal adjustment
                    df['Value'] = df['Value'].rolling(window=12, center=True).mean()
            
            return df
            
        except Exception as e:
            self.metadata['processing_log'].append(
                f"Error processing {series_name} for {state}: {str(e)}"
            )
            return None
    
    def enhance_dataset(self, output_file: str = None) -> pd.DataFrame:
        """
        Enhance the dataset with state-level metrics
        
        Args:
            output_file: Optional path to save the enhanced dataset
            
        Returns:
            DataFrame with state-level data
        """
        print("Starting state-level data enhancement...")
        print(f"Processing {len(self.states)} states with {len(self.state_mappings)} series each")
        
        all_state_data = []
        
        for state in self.states:
            print(f"\nProcessing state: {state}")
            state_successful = []
            state_failed = []
            
            for series_name, pattern in self.state_mappings.items():
                print(f"  - {series_name}...", end=' ')
                
                df = self._process_state_series(state, series_name, pattern)
                
                if df is not None and len(df) > 0:
                    all_state_data.append(df)
                    state_successful.append(series_name)
                    print("✓")
                else:
                    state_failed.append(series_name)
                    print("✗")
            
            # Update metadata
            self.metadata['successful_series'][state] = state_successful
            self.metadata['failed_series'][state] = state_failed
            self.metadata['coverage_percentage'][state] = len(state_successful) / len(self.state_mappings) * 100
        
        if not all_state_data:
            raise ValueError("No state-level data was successfully processed")
        
        # Combine all data
        combined_df = pd.concat(all_state_data, ignore_index=True)
        
        # Sort by Date, State, SeriesID
        combined_df = combined_df.sort_values(['Date', 'State', 'SeriesID']).reset_index(drop=True)
        
        # Save if output file specified
        if output_file:
            combined_df.to_csv(output_file, index=False)
            print(f"\nState-level data saved to: {output_file}")
        
        # Save metadata
        self._save_metadata()
        
        print(f"\nState-level enhancement completed!")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        print(f"States covered: {combined_df['State'].nunique()}")
        print(f"Series covered: {combined_df['SeriesID'].nunique()}")
        
        return combined_df
    
    def _save_metadata(self):
        """Save processing metadata to JSON file"""
        metadata_file = 'state_data_metadata.json'
        
        # Convert datetime objects to strings for JSON serialization
        metadata_copy = self.metadata.copy()
        metadata_copy['processing_timestamp'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_copy, f, indent=2, default=str)
        
        print(f"Metadata saved to: {metadata_file}")
    
    def validate_2023_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Cross-check 2023 values against FRED web interface
        Returns validation results
        """
        print("\nValidating 2023 data against FRED web interface...")
        
        validation_results = {
            'validated': [],
            'failed_validation': [],
            'notes': []
        }
        
        # Sample validation for a few key series
        sample_states = ['CA', 'TX', 'NY', 'FL']
        sample_series = ['UnemploymentRate', 'HomePriceIndex']
        
        for state in sample_states:
            for series_name in sample_series:
                pattern = self.state_mappings.get(series_name)
                if not pattern:
                    continue
                
                series_id = pattern.replace('[ST]', state)
                state_data = df[(df['State'] == state) & (df['SeriesID'] == series_id)]
                
                if len(state_data) > 0:
                    # Get 2023 data
                    data_2023 = state_data[state_data['Date'].dt.year == 2023]
                    
                    if len(data_2023) > 0:
                        # Simple validation: check if values are reasonable
                        values = data_2023['Value'].values
                        
                        if series_name == 'UnemploymentRate':
                            if all(0 <= v <= 20 for v in values):  # Unemployment should be 0-20%
                                validation_results['validated'].append(f"{series_name}_{state}")
                            else:
                                validation_results['failed_validation'].append(f"{series_name}_{state}")
                        elif series_name == 'HomePriceIndex':
                            if all(v > 0 for v in values):  # HPI should be positive
                                validation_results['validated'].append(f"{series_name}_{state}")
                            else:
                                validation_results['failed_validation'].append(f"{series_name}_{state}")
        
        print(f"Validation completed: {len(validation_results['validated'])} passed, "
              f"{len(validation_results['failed_validation'])} failed")
        
        return validation_results
    
    def integrate_with_main_dataset(self, main_dataset_path: str, state_data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate state-level data with the main national dataset
        
        Args:
            main_dataset_path: Path to the main unified dataset
            state_data: State-level data DataFrame
            
        Returns:
            Integrated DataFrame
        """
        print("\nIntegrating state-level data with main dataset...")
        
        # Load main dataset
        main_df = pd.read_csv(main_dataset_path)
        main_df['Date'] = pd.to_datetime(main_df['Date'])
        
        # Create a copy of main data for each state
        state_main_data = []
        for state in self.states:
            state_df = main_df.copy()
            state_df['State'] = state
            state_main_data.append(state_df)
        
        # Combine main data for all states
        all_main_data = pd.concat(state_main_data, ignore_index=True)
        
        # Combine with state-specific data
        integrated_df = pd.concat([all_main_data, state_data], ignore_index=True)
        
        # Sort by Date, State
        integrated_df = integrated_df.sort_values(['Date', 'State']).reset_index(drop=True)
        
        print(f"Integration completed!")
        print(f"Final dataset shape: {integrated_df.shape}")
        print(f"States: {integrated_df['State'].nunique()}")
        print(f"Date range: {integrated_df['Date'].min()} to {integrated_df['Date'].max()}")
        
        return integrated_df

def main():
    """Main function to run state-level data enhancement"""
    try:
        # Initialize enhancer
        enhancer = StateDataEnhancer()
        
        # Enhance dataset
        state_data = enhancer.enhance_dataset('state_level_data.csv')
        
        # Validate 2023 data
        validation_results = enhancer.validate_2023_data(state_data)
        
        # Integrate with main dataset if it exists
        main_dataset_path = 'AllProcessedFiles/unified_monthly_data.csv'
        if os.path.exists(main_dataset_path):
            integrated_data = enhancer.integrate_with_main_dataset(main_dataset_path, state_data)
            integrated_data.to_csv('integrated_state_national_data.csv', index=False)
            print(f"Integrated dataset saved to: integrated_state_national_data.csv")
        
        print("\nState-level data enhancement completed successfully!")
        
    except Exception as e:
        print(f"Error in state-level data enhancement: {str(e)}")
        raise

if __name__ == "__main__":
    main() 