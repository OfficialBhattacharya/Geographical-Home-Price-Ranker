import os
import sys
import time
from msaDataEnhancer import MSADataEnhancer
from config_loader import get_config

def run_batch_msa_scraping():
    """Run batch MSA-level data scraping"""
    
    print("MSA-Level FRED Data Scraping")
    print("=" * 60)
    
    try:
        # Validate configuration
        print("\nValidating configuration...")
        config = get_config()
        config.validate_config()
        print("[OK] Configuration validated successfully")
        
        # Check if MSA reference files exist
        if not os.path.exists('msa_list_top150.txt'):
            print("[ERROR] msa_list_top150.txt not found")
            print("Please ensure the MSA reference file is in the current directory")
            return False
            
        if not os.path.exists('msa_level_fred_mappings.txt'):
            print("[ERROR] msa_level_fred_mappings.txt not found")
            print("Please ensure the MSA mappings file is in the current directory")
            return False
        
        # Initialize MSA data enhancer
        print("\nInitializing MSA data enhancer...")
        enhancer = MSADataEnhancer(config)
        
        print(f"Loaded {len(enhancer.msa_reference)} MSAs")
        print(f"Loaded {len(enhancer.msa_mappings)} series mappings")
        
        # Show available MSA metrics
        print(f"\nAvailable MSA-level metrics:")
        for i, metric in enumerate(enhancer.msa_mappings.keys(), 1):
            print(f"  {i}. {metric}")
        
        # Ask user for processing preferences
        print("\nProcessing Options:")
        print("1. Process all 150 MSAs (full dataset)")
        print("2. Process top 50 MSAs only (faster)")
        print("3. Process top 10 MSAs only (testing)")
        print("4. Custom MSA count")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '2':
            print("Processing top 50 MSAs...")
            enhancer.msa_reference = enhancer.msa_reference.head(50)
        elif choice == '3':
            print("Processing top 10 MSAs...")
            enhancer.msa_reference = enhancer.msa_reference.head(10)
        elif choice == '4':
            try:
                count = int(input("Enter number of MSAs to process: ").strip())
                if 1 <= count <= len(enhancer.msa_reference):
                    print(f"Processing top {count} MSAs...")
                    enhancer.msa_reference = enhancer.msa_reference.head(count)
                else:
                    print(f"Invalid count. Using all {len(enhancer.msa_reference)} MSAs")
            except ValueError:
                print("Invalid input. Using all MSAs")
        else:
            print("Processing all 150 MSAs...")
        
        # Ask user for output preferences
        print("\nOutput Options:")
        print("1. MSA-level data only (msa_level_data.csv)")
        print("2. Integrated with existing data (integrated_national_state_msa_data.csv)")
        print("3. Both")
        
        output_choice = input("\nEnter your choice (1-3): ").strip()
        
        # Run MSA-level enhancement
        print(f"\nStarting MSA-level data enhancement for {len(enhancer.msa_reference)} MSAs...")
        print("This may take several minutes depending on the number of MSAs...")
        
        start_time = time.time()
        msa_data = enhancer.enhance_dataset('msa_level_data.csv')
        end_time = time.time()
        
        if len(msa_data) == 0:
            print("\n[ERROR] No MSA data was successfully processed")
            return False
        
        processing_time = end_time - start_time
        print(f"\n[OK] MSA data processing completed in {processing_time:.1f} seconds")
        print(f"Final dataset shape: {msa_data.shape}")
        
        # Validate data
        print("\nValidating 2023 data...")
        validation_results = enhancer.validate_2023_data(msa_data)
        
        # Integration with existing data
        if output_choice in ['2', '3']:
            print("\nChecking for existing datasets to integrate...")
            
            # Look for existing integrated data or state-level data
            possible_files = [
                'integrated_state_national_data.csv',
                'state_level_data.csv',
                'unified_monthly_data.csv'
            ]
            
            integration_file = None
            for filename in possible_files:
                if os.path.exists(filename):
                    integration_file = filename
                    break
            
            if integration_file:
                print(f"Integrating with {integration_file}...")
                integrated_data = enhancer.integrate_with_main_dataset(integration_file, msa_data)
                
                if len(integrated_data) > 0:
                    print(f"[OK] Integration successful")
                else:
                    print("[WARNING] Integration failed, but MSA data was saved separately")
            else:
                print("[WARNING] No existing dataset found for integration")
                print("MSA data saved as standalone file")
        
        # Generate summary report
        print("\n" + "="*60)
        print("MSA DATA PROCESSING SUMMARY")
        print("="*60)
        
        successful_msas = len(enhancer.metadata['successful_series'])
        total_msas = len(enhancer.msa_reference)
        
        print(f"MSAs Processed: {total_msas}")
        print(f"MSAs with Data: {successful_msas}")
        print(f"Success Rate: {(successful_msas/total_msas)*100:.1f}%")
        print(f"Total Data Points: {len(msa_data):,}")
        print(f"Quarterly Series Interpolated: {len(enhancer.metadata['quarterly_interpolated'])}")
        print(f"Low Coverage Series Skipped: {len(enhancer.metadata['low_coverage_skipped'])}")
        
        # Show coverage by series
        if msa_data is not None and len(msa_data) > 0:
            series_coverage = msa_data['SeriesName'].value_counts()
            print(f"\nSeries Coverage:")
            for series, count in series_coverage.items():
                msa_count = len(msa_data[msa_data['SeriesName'] == series]['CBSA_Code'].unique())
                print(f"  {series}: {msa_count}/{total_msas} MSAs ({(msa_count/total_msas)*100:.1f}%)")
        
        # Output file locations
        print(f"\nOutput Files Created:")
        processed_dir = config.get_processed_data_dir()
        
        if output_choice in ['1', '3']:
            msa_file = os.path.join(processed_dir, 'msa_level_data.csv')
            if os.path.exists(msa_file):
                file_size = os.path.getsize(msa_file) / (1024*1024)  # MB
                print(f"  - MSA data: {msa_file} ({file_size:.1f} MB)")
        
        if output_choice in ['2', '3'] and os.path.exists(os.path.join(processed_dir, 'integrated_national_state_msa_data.csv')):
            integrated_file = os.path.join(processed_dir, 'integrated_national_state_msa_data.csv')
            file_size = os.path.getsize(integrated_file) / (1024*1024)  # MB
            print(f"  - Integrated data: {integrated_file} ({file_size:.1f} MB)")
        
        metadata_file = os.path.join(processed_dir, 'msa_data_metadata.json')
        if os.path.exists(metadata_file):
            print(f"  - Metadata: {metadata_file}")
        
        print(f"\n[SUCCESS] MSA-level data enhancement completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] MSA-level data scraping failed: {str(e)}")
        print("\nPlease check:")
        print("1. FRED API key is set in config.yaml")
        print("2. msa_list_top150.txt and msa_level_fred_mappings.txt exist")
        print("3. Internet connection is available")
        print("4. Sufficient disk space for output files")
        return False

def show_msa_help():
    """Show help information for MSA data processing"""
    
    print("""
MSA-Level FRED Data Processing Help

This script extends the FRED data pipeline to include Metropolitan Statistical Area (MSA)
level economic indicators for the top 150 metropolitan areas in the United States.

Available MSA-Level Metrics:
- UnemploymentRate: Monthly unemployment rates by MSA
- HomePriceIndex: FHFA All-Transactions Home Price Index (quarterly, interpolated to monthly)
- MedianDaysonMarket: Median days properties spend on market (Realtor.com data)
- MedianListingPriceperSquareFeet: Median listing price per square foot (Realtor.com data)

Coverage Notes:
- Unemployment data: Available for most MSAs
- Home Price Index: Available for major MSAs, quarterly data interpolated to monthly
- Realtor.com metrics: Limited coverage, varies by MSA size and region
- Minimum 60% data coverage required (last 2 years)

Prerequisites:
- config.yaml with FRED API key
- msa_list_top150.txt with CBSA codes and MSA names
- msa_level_fred_mappings.txt with FRED series patterns
- Internet connection for API calls

Usage:
  python batch_msa_scraper.py              # Interactive mode
  python batch_msa_scraper.py --help       # Show this help

Output Files:
- msa_level_data.csv: MSA-specific economic data
- msa_data_metadata.json: Processing metadata and coverage information
- integrated_national_state_msa_data.csv: Combined national/state/MSA dataset

For more information, see the main README.md
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_msa_help()
        sys.exit(0)
    
    print("MSA-Level FRED Data Enhancement")
    print("=" * 50)
    
    # Check prerequisites first
    if not os.path.exists('msa_list_top150.txt'):
        print("[ERROR] msa_list_top150.txt not found!")
        print("Please ensure the MSA reference file exists in the current directory.")
        sys.exit(1)
        
    if not os.path.exists('msa_level_fred_mappings.txt'):
        print("[ERROR] msa_level_fred_mappings.txt not found!")
        print("Please ensure the MSA mappings file exists in the current directory.")
        sys.exit(1)
    
    success = run_batch_msa_scraping()
    
    if success:
        print("\n[SUCCESS] MSA data processing completed successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] MSA data processing failed!")
        sys.exit(1) 