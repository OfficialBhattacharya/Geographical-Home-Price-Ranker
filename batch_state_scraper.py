import os
import sys
import time
from stateDataEnhancer import StateDataEnhancer
from config_loader import get_config

def run_batch_state_scraping():
    """Run batch state-level data scraping"""
    
    print("State-Level FRED Data Scraping")
    print("=" * 60)
    
    try:
        # Validate configuration
        print("\nValidating configuration...")
        config = get_config()
        config.validate_config()
        print("[OK] Configuration validated successfully")
        
        # Check if state mappings file exists
        if not os.path.exists('state_level_fred_mappings.txt'):
            print("[ERROR] state_level_fred_mappings.txt not found")
            print("Please ensure the state mappings file is in the current directory")
            return False
        
        # Initialize state data enhancer
        print("\nInitializing state data enhancer...")
        enhancer = StateDataEnhancer(config)
        
        print(f"Loaded {len(enhancer.states)} states")
        print(f"Loaded {len(enhancer.state_mappings)} series mappings")
        
        # Ask user for output preferences
        print("\nOutput Options:")
        print("1. State-level data only (state_level_data.csv)")
        print("2. Integrated with national data (integrated_state_national_data.csv)")
        print("3. Both")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        # Run state-level enhancement
        print("\nStarting state-level data enhancement...")
        state_data = enhancer.enhance_dataset('state_level_data.csv')
        
        # Validate data
        print("\nValidating 2023 data...")
        validation_results = enhancer.validate_2023_data(state_data)
        
        # Handle integration based on user choice
        if choice in ['2', '3']:
            main_dataset_path = config.get_unified_data_path()
            if os.path.exists(main_dataset_path):
                print("\nIntegrating with main dataset...")
                integrated_data = enhancer.integrate_with_main_dataset(main_dataset_path, state_data)
                integrated_data.to_csv('integrated_state_national_data.csv', index=False)
                print(f"Integrated dataset saved to: integrated_state_national_data.csv")
            else:
                print(f"\n[WARNING] Main dataset not found at: {main_dataset_path}")
                print("Skipping integration step")
        
        # Show summary
        print(f"\n{'='*60}")
        print("STATE-LEVEL DATA SCRAPING COMPLETED!")
        print(f"{'='*60}")
        
        print(f"\nSummary:")
        print(f"  - States processed: {len(enhancer.states)}")
        print(f"  - Series processed: {len(enhancer.state_mappings)}")
        print(f"  - Total records: {len(state_data)}")
        print(f"  - Date range: {state_data['Date'].min()} to {state_data['Date'].max()}")
        
        print(f"\nCoverage Summary:")
        coverage_stats = enhancer.metadata['coverage_percentage']
        avg_coverage = sum(coverage_stats.values()) / len(coverage_stats)
        print(f"  - Average coverage: {avg_coverage:.1f}%")
        print(f"  - States with >80% coverage: {sum(1 for v in coverage_stats.values() if v > 80)}")
        print(f"  - States with <50% coverage: {sum(1 for v in coverage_stats.values() if v < 50)}")
        
        print(f"\nValidation Results:")
        print(f"  - Validated series: {len(validation_results['validated'])}")
        print(f"  - Failed validation: {len(validation_results['failed_validation'])}")
        
        print(f"\nOutput Files:")
        if choice in ['1', '3']:
            print(f"  - State-level data: state_level_data.csv")
        if choice in ['2', '3'] and os.path.exists('integrated_state_national_data.csv'):
            print(f"  - Integrated data: integrated_state_national_data.csv")
        print(f"  - Metadata: state_data_metadata.json")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] State-level data scraping failed: {str(e)}")
        print("\nPlease check:")
        print("1. FRED API key is set in config.yaml")
        print("2. state_level_fred_mappings.txt exists")
        print("3. Internet connection is available")
        return False

def show_help():
    """Show help information"""
    
    print("""
State-Level FRED Data Scraping Help

This script fetches and processes state-level economic indicators from the FRED API.

Prerequisites:
- config.yaml file with your FRED API key
- state_level_fred_mappings.txt file with state series mappings
- Required Python packages: pandas, numpy, requests, pyyaml

Usage:
  python batch_state_scraper.py [OPTIONS]

Options:
  --help, -h     Show this help message
  --check, -c    Check requirements only (don't run scraping)

Examples:
  python batch_state_scraper.py              # Run complete state scraping
  python batch_state_scraper.py --check      # Check requirements

Output:
- state_level_data.csv: State-level economic indicators
- integrated_state_national_data.csv: Combined state and national data
- state_data_metadata.json: Processing metadata and coverage statistics

For more information, see README.md
""")

def check_requirements():
    """Check if all requirements are met before running state scraping"""
    
    print("Checking state-level scraping requirements...")
    
    # Check if required files exist
    required_files = [
        "config.yaml",
        "state_level_fred_mappings.txt", 
        "stateDataEnhancer.py",
        "config_loader.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("[ERROR] Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check Python packages
    try:
        import pandas
        import numpy
        import requests
        import yaml
    except ImportError as e:
        print(f"[ERROR] Missing required Python package: {e}")
        print("Please install with: pip install pandas numpy requests pyyaml")
        return False
    
    # Test configuration
    try:
        config = get_config()
        config.validate_config()
        print("[OK] Configuration test successful")
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False
    
    print("[OK] All requirements satisfied")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
            sys.exit(0)
        elif arg in ['--check', '-c']:
            success = check_requirements()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Run the batch state scraping
    success = run_batch_state_scraping()
    sys.exit(0 if success else 1) 