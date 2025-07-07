#!/usr/bin/env python3
"""
Test script for state-level data enhancement functionality
"""

import os
import sys
import pandas as pd
from datetime import datetime
from stateDataEnhancer import StateDataEnhancer
from config_loader import get_config

def test_state_mappings_loading():
    """Test that state mappings are loaded correctly"""
    print("Testing state mappings loading...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Check that mappings were loaded
        assert len(enhancer.state_mappings) > 0, "No state mappings loaded"
        print(f"✓ Loaded {len(enhancer.state_mappings)} state mappings")
        
        # Check that states were loaded
        assert len(enhancer.states) == 51, f"Expected 51 states, got {len(enhancer.states)}"
        print(f"✓ Loaded {len(enhancer.states)} states")
        
        # Check for specific mappings
        expected_mappings = [
            'UnemploymentRate',
            'HomePriceIndex', 
            'AverageSalesPrice_NewHousesSold'
        ]
        
        for mapping in expected_mappings:
            assert mapping in enhancer.state_mappings, f"Missing mapping: {mapping}"
        
        print("✓ All expected mappings present")
        return True
        
    except Exception as e:
        print(f"✗ State mappings test failed: {e}")
        return False

def test_fred_api_connection():
    """Test FRED API connection with a simple series"""
    print("\nTesting FRED API connection...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Test with a simple, reliable series (national unemployment)
        test_series = "UNRATE"
        df = enhancer._fetch_fred_series(test_series)
        
        if df is not None and len(df) > 0:
            print(f"✓ Successfully fetched {len(df)} records for {test_series}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            return True
        else:
            print(f"✗ Failed to fetch data for {test_series}")
            return False
            
    except Exception as e:
        print(f"✗ FRED API test failed: {e}")
        return False

def test_state_series_processing():
    """Test processing a single state-series combination"""
    print("\nTesting state series processing...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Test with a simple state and series
        test_state = "CA"
        test_series = "UnemploymentRate"
        pattern = enhancer.state_mappings.get(test_series)
        
        if not pattern:
            print(f"✗ No pattern found for {test_series}")
            return False
        
        print(f"Testing {test_series} for {test_state} (pattern: {pattern})")
        
        df = enhancer._process_state_series(test_state, test_series, pattern)
        
        if df is not None and len(df) > 0:
            print(f"✓ Successfully processed {len(df)} records")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"  State: {df['State'].iloc[0]}")
            print(f"  SeriesID: {df['SeriesID'].iloc[0]}")
            return True
        else:
            print(f"✗ Failed to process {test_series} for {test_state}")
            return False
            
    except Exception as e:
        print(f"✗ State series processing test failed: {e}")
        return False

def test_frequency_detection():
    """Test frequency detection functionality"""
    print("\nTesting frequency detection...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Create test data with known frequencies
        monthly_dates = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
        quarterly_dates = pd.date_range('2023-01-01', '2023-12-01', freq='Q')
        
        monthly_df = pd.DataFrame({'Date': monthly_dates, 'Value': range(len(monthly_dates))})
        quarterly_df = pd.DataFrame({'Date': quarterly_dates, 'Value': range(len(quarterly_dates))})
        
        # Test frequency detection
        monthly_freq = enhancer._determine_frequency(monthly_df)
        quarterly_freq = enhancer._determine_frequency(quarterly_df)
        
        print(f"Monthly data frequency: {monthly_freq}")
        print(f"Quarterly data frequency: {quarterly_freq}")
        
        assert monthly_freq == 'monthly', f"Expected monthly, got {monthly_freq}"
        assert quarterly_freq == 'quarterly', f"Expected quarterly, got {quarterly_freq}"
        
        print("✓ Frequency detection working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Frequency detection test failed: {e}")
        return False

def test_interpolation():
    """Test quarterly to monthly interpolation"""
    print("\nTesting interpolation...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Create quarterly test data
        quarterly_dates = pd.date_range('2023-01-01', '2023-12-01', freq='Q')
        quarterly_df = pd.DataFrame({
            'Date': quarterly_dates,
            'Value': [100, 110, 120, 130]
        })
        
        # Test interpolation
        interpolated_df = enhancer._interpolate_quarterly_to_monthly(quarterly_df)
        
        print(f"Original quarterly records: {len(quarterly_df)}")
        print(f"Interpolated monthly records: {len(interpolated_df)}")
        
        # Should have 12 monthly records
        assert len(interpolated_df) == 12, f"Expected 12 records, got {len(interpolated_df)}"
        
        # Check that values are interpolated
        assert not interpolated_df['Value'].isna().any(), "Interpolated data contains NaN values"
        
        print("✓ Interpolation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Interpolation test failed: {e}")
        return False

def test_metadata_tracking():
    """Test metadata tracking functionality"""
    print("\nTesting metadata tracking...")
    
    try:
        enhancer = StateDataEnhancer()
        
        # Check initial metadata structure
        assert 'successful_series' in enhancer.metadata
        assert 'failed_series' in enhancer.metadata
        assert 'coverage_percentage' in enhancer.metadata
        assert 'processing_log' in enhancer.metadata
        
        # Test adding to processing log
        test_message = "Test processing message"
        enhancer.metadata['processing_log'].append(test_message)
        
        assert test_message in enhancer.metadata['processing_log']
        
        print("✓ Metadata tracking working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Metadata tracking test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("State-Level Data Enhancement Test Suite")
    print("=" * 50)
    
    tests = [
        test_state_mappings_loading,
        test_fred_api_connection,
        test_state_series_processing,
        test_frequency_detection,
        test_interpolation,
        test_metadata_tracking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! State-level enhancement is ready to use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False

def test_small_enhancement():
    """Test a small-scale enhancement with limited states"""
    print("\n" + "="*50)
    print("Testing Small-Scale Enhancement")
    print("="*50)
    
    try:
        enhancer = StateDataEnhancer()
        
        # Limit to just 2 states for testing
        test_states = ['CA', 'TX']
        enhancer.states = test_states
        
        print(f"Testing with states: {test_states}")
        
        # Run enhancement
        state_data = enhancer.enhance_dataset('test_state_data.csv')
        
        if state_data is not None and len(state_data) > 0:
            print(f"✓ Successfully created test dataset with {len(state_data)} records")
            print(f"  States: {state_data['State'].unique()}")
            print(f"  Series: {state_data['SeriesID'].nunique()}")
            print(f"  Date range: {state_data['Date'].min()} to {state_data['Date'].max()}")
            
            # Clean up test file
            if os.path.exists('test_state_data.csv'):
                os.remove('test_state_data.csv')
                print("  Test file cleaned up")
            
            return True
        else:
            print("✗ Failed to create test dataset")
            return False
            
    except Exception as e:
        print(f"✗ Small-scale enhancement test failed: {e}")
        return False

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        "state_level_fred_mappings.txt",
        "config.yaml",
        "stateDataEnhancer.py",
        "config_loader.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all required files are present before running tests.")
        sys.exit(1)
    
    # Run tests
    basic_tests_passed = run_all_tests()
    
    if basic_tests_passed:
        # Only run enhancement test if basic tests pass
        enhancement_test_passed = test_small_enhancement()
        
        if enhancement_test_passed:
            print("\n🎉 All tests completed successfully!")
            print("State-level data enhancement is ready for production use.")
            sys.exit(0)
        else:
            print("\n⚠️  Enhancement test failed. Check configuration and API key.")
            sys.exit(1)
    else:
        print("\n❌ Basic tests failed. Please fix issues before proceeding.")
        sys.exit(1) 