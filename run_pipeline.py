import subprocess
import sys
import os
from config_loader import get_config

def run_pipeline():
    """Run the complete FRED data processing pipeline"""
    
    print("FRED Data Processing Pipeline")
    print("=" * 60)
    
    try:
        # First, validate configuration
        print("\nValidating configuration...")
        config = get_config()
        config.validate_config()
        print("[OK] Configuration validated successfully")
        
        # Define pipeline steps
        steps = [
            {
                "name": "Downloading FRED data", 
                "command": ["python", "batch_fred_scraper.py"],
                "description": "Downloads all FRED series from allFredSeries.txt"
            },
            {
                "name": "Aggregating data", 
                "command": ["python", "fredAggregator.py"],
                "description": "Combines all CSV files into unified monthly dataset"
            },
            {
                "name": "Cleaning and interpolating", 
                "command": ["python", "fredCleaner.py"],
                "description": "Fills missing values and extends dataset to end date"
            },
            {
                "name": "State-level data enhancement", 
                "command": ["python", "stateDataEnhancer.py"],
                "description": "Fetches and processes state-level economic indicators"
            },
            {
                "name": "MSA-level data enhancement", 
                "command": ["python", "batch_msa_scraper.py"],
                "description": "Fetches and processes MSA-level economic indicators for top 150 metropolitan areas"
            }
        ]
        
        # Track progress
        completed_steps = 0
        total_steps = len(steps)
        
        for i, step in enumerate(steps, 1):
            print(f"\n{'='*60}")
            print(f"STEP {i}/{total_steps}: {step['name']}")
            print(f"{'='*60}")
            print(f"Description: {step['description']}")
            print()
            
            try:
                # Run the step
                result = subprocess.run(
                    step["command"], 
                    check=True,
                    text=True,
                    capture_output=False  # Show real-time output
                )
                
                print(f"\n[SUCCESS] {step['name']} completed successfully")
                completed_steps += 1
                
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] {step['name']} failed with return code: {e.returncode}")
                print("Check the output above for error details.")
                
                # Ask user if they want to continue
                if i < total_steps:
                    response = input(f"\nDo you want to continue with the remaining {total_steps - i} steps? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Pipeline stopped by user.")
                        return False
                else:
                    return False
                    
            except KeyboardInterrupt:
                print(f"\n[WARNING] Pipeline interrupted by user during: {step['name']}")
                return False
            except Exception as e:
                print(f"\n[ERROR] Unexpected error during {step['name']}: {str(e)}")
                return False
        
        # Final summary
        print(f"\n{'='*60}")
        if completed_steps == total_steps:
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"[OK] All {total_steps} steps completed")
            
            # Show output file locations
            print(f"\nOutput Files:")
            print(f"   Raw data: {config.get_raw_data_dir()}")
            print(f"   Processed data: {config.get_processed_data_dir()}")
            print(f"   Unified dataset: {config.get_unified_data_path()}")
            
        else:
            print(f"PIPELINE PARTIALLY COMPLETED")
            print(f"[OK] {completed_steps}/{total_steps} steps completed")
            
        print(f"{'='*60}")
        return completed_steps == total_steps
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline setup error: {str(e)}")
        print("\nPlease check:")
        print("1. config.yaml exists and is properly formatted")
        print("2. All required Python packages are installed")
        print("3. FRED API key is set in config.yaml")
        return False

def check_requirements():
    """Check if all requirements are met before running pipeline"""
    
    print("Checking requirements...")
    
    # Check if required files exist
    required_files = [
        "config.yaml",
        "allFredSeries.txt", 
        "fredScraper.py",
        "batch_fred_scraper.py", 
        "fredAggregator.py",
        "fredCleaner.py",
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
    
    print("[OK] All requirements satisfied")
    return True

def show_help():
    """Show help information"""
    
    print("""
FRED Data Processing Pipeline Help

This script runs the complete FRED data processing pipeline in sequence:

1. Downloads FRED series data from the Federal Reserve API
2. Aggregates all CSV files into a unified monthly dataset  
3. Cleans and interpolates missing values
4. Enhances dataset with state-level economic indicators
5. Enhances dataset with MSA-level economic indicators (top 150 metros)

Prerequisites:
- config.yaml file with your FRED API key and paths
- allFredSeries.txt file with FRED series to download
- state_level_fred_mappings.txt for state-level series patterns
- msa_list_top150.txt and msa_level_fred_mappings.txt for MSA-level processing
- Required Python packages: pandas, numpy, requests, pyyaml

Usage:
  python run_pipeline.py [OPTIONS]

Options:
  --help, -h     Show this help message
  --check, -c    Check requirements only (don't run pipeline)
  --config, -cfg Test configuration only

Examples:
  python run_pipeline.py              # Run complete pipeline
  python run_pipeline.py --check      # Check requirements
  python run_pipeline.py --config     # Test configuration

For more information, see README.md
""")

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
        elif arg in ['--config', '-cfg']:
            try:
                config = get_config()
                config.validate_config()
                print("[OK] Configuration test successful!")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] Configuration test failed: {e}")
                sys.exit(1)
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Run the pipeline
    if not check_requirements():
        print("\n[ERROR] Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    success = run_pipeline()
    sys.exit(0 if success else 1) 