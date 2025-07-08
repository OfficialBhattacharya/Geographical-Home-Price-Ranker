import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from msaDataEnhancer import MSADataEnhancer
from config_loader import get_config
import warnings
warnings.filterwarnings('ignore')

class MSAEnhancementValidator:
    """
    Comprehensive validation tests for MSA data enhancement functionality
    """
    
    def __init__(self):
        self.config = get_config()
        self.test_results = {
            'file_validation': {},
            'api_connectivity': {},
            'data_processing': {},
            'spot_checks': {},
            'integration_tests': {},
            'summary': {}
        }
        
    def validate_prerequisites(self):
        """Test that all prerequisite files exist and are properly formatted"""
        print("\n=== PREREQUISITE VALIDATION ===")
        
        # Check MSA reference file
        try:
            if os.path.exists('msa_list_top150.txt'):
                with open('msa_list_top150.txt', 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                
                # Validate format
                valid_entries = 0
                for line in lines:
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2 and parts[0].strip().isdigit() and len(parts[0].strip()) == 5:
                            valid_entries += 1
                
                self.test_results['file_validation']['msa_reference'] = {
                    'exists': True,
                    'total_entries': len(lines),
                    'valid_entries': valid_entries,
                    'format_valid': valid_entries > 100  # Expect at least 100 valid MSAs
                }
                print(f"✓ msa_list_top150.txt: {valid_entries}/{len(lines)} valid entries")
            else:
                self.test_results['file_validation']['msa_reference'] = {'exists': False}
                print("✗ msa_list_top150.txt not found")
                
        except Exception as e:
            self.test_results['file_validation']['msa_reference'] = {'exists': True, 'error': str(e)}
            print(f"✗ Error reading msa_list_top150.txt: {e}")
        
        # Check MSA mappings file
        try:
            if os.path.exists('msa_level_fred_mappings.txt'):
                with open('msa_level_fred_mappings.txt', 'r') as f:
                    lines = f.readlines()
                
                # Count valid mappings
                valid_mappings = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2 and 'NO_MSA_EQUIV' not in parts[1]:
                            valid_mappings += 1
                
                self.test_results['file_validation']['msa_mappings'] = {
                    'exists': True,
                    'valid_mappings': valid_mappings,
                    'format_valid': valid_mappings >= 3  # Expect at least 3 mappable series
                }
                print(f"✓ msa_level_fred_mappings.txt: {valid_mappings} valid mappings")
            else:
                self.test_results['file_validation']['msa_mappings'] = {'exists': False}
                print("✗ msa_level_fred_mappings.txt not found")
                
        except Exception as e:
            self.test_results['file_validation']['msa_mappings'] = {'exists': True, 'error': str(e)}
            print(f"✗ Error reading msa_level_fred_mappings.txt: {e}")
    
    def test_fred_api_connection(self):
        """Test FRED API connection with MSA-specific series"""
        print("\n=== FRED API CONNECTIVITY TEST ===")
        
        try:
            enhancer = MSADataEnhancer()
            
            # Test series for major MSAs
            test_cases = [
                ('35620', 'UnemploymentRate', '35620UR'),  # NYC unemployment
                ('31080', 'HomePriceIndex', 'ATNHPIUS31080Q'),  # LA home price index
                ('16980', 'UnemploymentRate', '16980UR')   # Chicago unemployment
            ]
            
            for cbsa, series_name, expected_id in test_cases:
                print(f"Testing {series_name} for CBSA {cbsa}...")
                
                df = enhancer._fetch_fred_series(expected_id)
                
                if df is not None and len(df) > 0:
                    recent_data = df[df['Date'] >= datetime.now() - timedelta(days=365)]
                    self.test_results['api_connectivity'][f"{cbsa}_{series_name}"] = {
                        'success': True,
                        'total_records': len(df),
                        'recent_records': len(recent_data),
                        'date_range': f"{df['Date'].min()} to {df['Date'].max()}"
                    }
                    print(f"  ✓ Retrieved {len(df)} records, {len(recent_data)} recent")
                else:
                    self.test_results['api_connectivity'][f"{cbsa}_{series_name}"] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    print(f"  ✗ No data retrieved")
                    
        except Exception as e:
            self.test_results['api_connectivity']['error'] = str(e)
            print(f"✗ API connectivity test failed: {e}")
    
    def test_data_processing_pipeline(self):
        """Test the complete MSA data processing pipeline with a small subset"""
        print("\n=== DATA PROCESSING PIPELINE TEST ===")
        
        try:
            enhancer = MSADataEnhancer()
            
            # Test with top 5 MSAs only
            original_reference = enhancer.msa_reference.copy()
            enhancer.msa_reference = enhancer.msa_reference.head(5)
            
            print(f"Testing with {len(enhancer.msa_reference)} MSAs...")
            print("MSAs being tested:")
            for _, row in enhancer.msa_reference.iterrows():
                print(f"  - {row['MSA_Name']} ({row['CBSA']})")
            
            # Run enhancement
            test_data = enhancer.enhance_dataset('test_msa_validation.csv')
            
            if len(test_data) > 0:
                # Analyze results
                unique_msas = test_data['CBSA_Code'].nunique()
                unique_series = test_data['SeriesName'].nunique()
                date_range = (test_data['Date'].min(), test_data['Date'].max())
                
                # Check for quarterly interpolation
                interpolated_data = test_data[test_data['Frequency'] == 'monthly_interpolated']
                
                self.test_results['data_processing'] = {
                    'success': True,
                    'total_records': len(test_data),
                    'unique_msas': unique_msas,
                    'unique_series': unique_series,
                    'date_range': date_range,
                    'interpolated_records': len(interpolated_data),
                    'series_coverage': dict(test_data['SeriesName'].value_counts())
                }
                
                print(f"✓ Processing successful:")
                print(f"  - Total records: {len(test_data):,}")
                print(f"  - MSAs with data: {unique_msas}/5")
                print(f"  - Series available: {unique_series}")
                print(f"  - Date range: {date_range[0]} to {date_range[1]}")
                print(f"  - Interpolated records: {len(interpolated_data)}")
                
                # Cleanup test file
                test_file = os.path.join(self.config.get_processed_data_dir(), 'test_msa_validation.csv')
                if os.path.exists(test_file):
                    os.remove(test_file)
                    
            else:
                self.test_results['data_processing'] = {
                    'success': False,
                    'error': 'No data processed'
                }
                print("✗ No data was processed")
                
            # Restore original reference
            enhancer.msa_reference = original_reference
            
        except Exception as e:
            self.test_results['data_processing'] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ Data processing test failed: {e}")
    
    def spot_check_major_msas(self):
        """Spot-check specific MSAs across different size tiers"""
        print("\n=== SPOT CHECK VALIDATION ===")
        
        # Define MSAs to spot-check by size tier
        test_msas = {
            'large': [
                ('35620', 'New York-Newark-Jersey City, NY-NJ-PA'),
                ('31080', 'Los Angeles-Long Beach-Anaheim, CA'),
                ('16980', 'Chicago-Naperville-Elgin, IL-IN-WI')
            ],
            'medium': [
                ('12420', 'Austin-Round Rock-Georgetown, TX'),
                ('16740', 'Charlotte-Concord-Gastonia, NC-SC'),
                ('19740', 'Denver-Aurora-Lakewood, CO')
            ],
            'smaller': [
                ('29620', 'Laredo, TX'),
                ('22180', 'Fargo, ND-MN'),
                ('23540', 'Gainesville, FL')
            ]
        }
        
        try:
            enhancer = MSADataEnhancer()
            
            for size_tier, msas in test_msas.items():
                print(f"\nTesting {size_tier} MSAs:")
                tier_results = {}
                
                for cbsa, name in msas:
                    print(f"  Checking {name} ({cbsa})...")
                    msa_results = {}
                    
                    # Test each available series
                    for series_name, pattern in enhancer.msa_mappings.items():
                        series_id = pattern.replace('[CBSA]', cbsa)
                        df = enhancer._fetch_fred_series(series_id)
                        
                        if df is not None and len(df) > 0:
                            # Check recent data availability
                            recent_date = datetime.now() - timedelta(days=365)
                            recent_data = df[df['Date'] >= recent_date]
                            
                            msa_results[series_name] = {
                                'available': True,
                                'total_records': len(df),
                                'recent_records': len(recent_data),
                                'latest_date': df['Date'].max().strftime('%Y-%m-%d')
                            }
                            print(f"    ✓ {series_name}: {len(df)} records, latest: {df['Date'].max().strftime('%Y-%m-%d')}")
                        else:
                            msa_results[series_name] = {
                                'available': False,
                                'error': 'No data'
                            }
                            print(f"    ✗ {series_name}: No data")
                    
                    tier_results[cbsa] = {
                        'name': name,
                        'series_results': msa_results,
                        'available_series': sum(1 for r in msa_results.values() if r.get('available', False)),
                        'total_series': len(msa_results)
                    }
                
                self.test_results['spot_checks'][size_tier] = tier_results
                
        except Exception as e:
            self.test_results['spot_checks']['error'] = str(e)
            print(f"✗ Spot check validation failed: {e}")
    
    def test_data_integration(self):
        """Test integration of MSA data with existing datasets"""
        print("\n=== DATA INTEGRATION TEST ===")
        
        try:
            # Look for existing datasets to test integration
            test_files = [
                'state_level_data.csv',
                'integrated_state_national_data.csv'
            ]
            
            integration_file = None
            for filename in test_files:
                if os.path.exists(filename):
                    integration_file = filename
                    break
            
            if integration_file:
                print(f"Testing integration with {integration_file}...")
                
                # Create small test MSA dataset
                enhancer = MSADataEnhancer()
                enhancer.msa_reference = enhancer.msa_reference.head(3)  # Small test
                
                # Mock MSA data for integration test
                test_msa_data = pd.DataFrame({
                    'Date': pd.date_range('2023-01-01', periods=12, freq='MS'),
                    'SeriesName': ['UnemploymentRate'] * 12,
                    'Value': np.random.uniform(3.0, 8.0, 12),
                    'CBSA_Code': ['35620'] * 12,
                    'MSA_Name': ['New York-Newark-Jersey City, NY-NJ-PA'] * 12,
                    'GeographyType': ['MSA'] * 12,
                    'GeographyCode': ['35620'] * 12,
                    'SeriesID': ['35620UR'] * 12,
                    'Source': ['FRED'] * 12
                })
                
                # Test integration
                integrated_data = enhancer.integrate_with_main_dataset(integration_file, test_msa_data)
                
                if len(integrated_data) > 0:
                    geography_types = integrated_data['GeographyType'].value_counts()
                    
                    self.test_results['integration_tests'] = {
                        'success': True,
                        'integration_file': integration_file,
                        'integrated_records': len(integrated_data),
                        'geography_breakdown': dict(geography_types)
                    }
                    
                    print(f"✓ Integration successful:")
                    print(f"  - Total integrated records: {len(integrated_data):,}")
                    print(f"  - Geography types: {dict(geography_types)}")
                else:
                    self.test_results['integration_tests'] = {
                        'success': False,
                        'error': 'Integration returned empty dataset'
                    }
                    print("✗ Integration failed - empty result")
                    
            else:
                self.test_results['integration_tests'] = {
                    'success': False,
                    'error': 'No existing dataset found for integration test'
                }
                print("⚠ No existing dataset found for integration testing")
                
        except Exception as e:
            self.test_results['integration_tests'] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ Integration test failed: {e}")
    
    def generate_summary_report(self):
        """Generate and display comprehensive test summary"""
        print("\n" + "="*60)
        print("MSA ENHANCEMENT VALIDATION SUMMARY")
        print("="*60)
        
        # Count successful tests
        successful_tests = 0
        total_tests = 0
        
        # File validation
        file_tests = self.test_results.get('file_validation', {})
        for test_name, result in file_tests.items():
            total_tests += 1
            if result.get('exists', False) and result.get('format_valid', False):
                successful_tests += 1
        
        # API connectivity
        api_tests = self.test_results.get('api_connectivity', {})
        api_success_count = sum(1 for result in api_tests.values() 
                               if isinstance(result, dict) and result.get('success', False))
        total_tests += len([k for k in api_tests.keys() if k != 'error'])
        successful_tests += api_success_count
        
        # Data processing
        if self.test_results.get('data_processing', {}).get('success', False):
            successful_tests += 1
        total_tests += 1
        
        # Integration test
        if self.test_results.get('integration_tests', {}).get('success', False):
            successful_tests += 1
        total_tests += 1
        
        # Overall success rate
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Test Status: {'PASS' if success_rate >= 80 else 'FAIL'}")
        
        # Detailed breakdown
        print(f"\nDetailed Results:")
        
        # File validation results
        file_results = self.test_results.get('file_validation', {})
        msa_ref = file_results.get('msa_reference', {})
        msa_map = file_results.get('msa_mappings', {})
        
        print(f"  File Validation:")
        print(f"    - MSA Reference: {'✓' if msa_ref.get('format_valid', False) else '✗'}")
        print(f"    - MSA Mappings: {'✓' if msa_map.get('format_valid', False) else '✗'}")
        
        # API connectivity results
        print(f"  API Connectivity: {api_success_count}/{len([k for k in api_tests.keys() if k != 'error'])} tests passed")
        
        # Data processing results
        processing_success = self.test_results.get('data_processing', {}).get('success', False)
        print(f"  Data Processing: {'✓' if processing_success else '✗'}")
        
        # Integration results
        integration_success = self.test_results.get('integration_tests', {}).get('success', False)
        print(f"  Data Integration: {'✓' if integration_success else '✗'}")
        
        # Spot check summary
        spot_checks = self.test_results.get('spot_checks', {})
        if spot_checks and 'error' not in spot_checks:
            print(f"  Spot Checks:")
            for tier, results in spot_checks.items():
                if isinstance(results, dict):
                    total_msas = len(results)
                    msas_with_data = sum(1 for msa_data in results.values() 
                                       if msa_data.get('available_series', 0) > 0)
                    print(f"    - {tier.title()} MSAs: {msas_with_data}/{total_msas} have data")
        
        print(f"\nValidation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save detailed results
        results_file = 'msa_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"Detailed results saved to: {results_file}")
        
        return success_rate >= 80

def run_msa_validation():
    """Run complete MSA enhancement validation suite"""
    
    print("MSA Data Enhancement Validation Suite")
    print("=" * 50)
    print(f"Starting validation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validator = MSAEnhancementValidator()
    
    try:
        # Run all validation tests
        validator.validate_prerequisites()
        validator.test_fred_api_connection()
        validator.test_data_processing_pipeline()
        validator.spot_check_major_msas()
        validator.test_data_integration()
        
        # Generate summary
        success = validator.generate_summary_report()
        
        return success
        
    except Exception as e:
        print(f"\n[ERROR] Validation suite failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("MSA Enhancement Validation")
    print("=" * 50)
    
    # Check if config is available
    try:
        config = get_config()
        config.validate_config()
        print("[OK] Configuration validated")
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        print("Please ensure config.yaml exists with valid FRED API key")
        exit(1)
    
    # Run validation
    success = run_msa_validation()
    
    if success:
        print("\n✓ MSA Enhancement validation PASSED")
        exit(0)
    else:
        print("\n✗ MSA Enhancement validation FAILED")
        exit(1) 