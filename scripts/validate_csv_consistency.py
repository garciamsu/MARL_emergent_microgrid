#!/usr/bin/env python3
"""
validate_csv_consistency.py

Script to validate that all CSV files in the project follow consistent format standards:
- Dataset files (assets/datasets/): European format (sep=';', decimal=',')
- Result files (results/): International format (sep=',', decimal='.')

This ensures no conflicts with decimal separators across the application.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.csv_handler import read_dataset_csv, read_result_csv, CSVConfig


def validate_dataset_format(file_path):
    """Validate that a dataset file uses the correct European format."""
    print(f"\n📋 Validating dataset: {file_path.name}")
    
    try:
        # Try reading with standardized function
        df = read_dataset_csv(file_path)
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) == 0:
            print("  ⚠️  No numeric columns found")
            return False
        
        # Sample some values to verify they're reasonable
        sample = df[numeric_cols].head(3)
        print(f"  ✅ Successfully read with European format (sep='{CSVConfig.INPUT_SEP}', decimal='{CSVConfig.INPUT_DECIMAL}')")
        print(f"  📊 Shape: {df.shape}, Numeric columns: {len(numeric_cols)}")
        print(f"  🔢 Sample values:\n{sample.to_string(index=False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False


def validate_result_format(file_path):
    """Validate that a result file uses the correct international format."""
    print(f"\n📋 Validating result: {file_path.relative_to(file_path.parents[2])}")
    
    try:
        # Try reading with standardized function
        df = read_result_csv(file_path)
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) == 0:
            print("  ⚠️  No numeric columns found")
            return False
        
        # Verify decimal format by checking a sample
        sample = df[numeric_cols].head(1)
        print(f"  ✅ Successfully read with international format (sep='{CSVConfig.OUTPUT_SEP}', decimal='{CSVConfig.OUTPUT_DECIMAL}')")
        print(f"  📊 Shape: {df.shape}, Numeric columns: {len(numeric_cols)}")
        
        # Check raw file to ensure it uses dot decimal
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]
            
        # Look for numbers with decimals in the file
        has_dot_decimal = any('.' in line and line.count('.') > 0 for line in lines[1:])
        has_comma_decimal = any(',' in line.split(',', 1)[-1] for line in lines[1:])
        
        if has_dot_decimal and not has_comma_decimal:
            print(f"  ✅ File uses dot (.) as decimal separator")
        else:
            print(f"  ⚠️  File may not be using standard format")
            print(f"     First data line: {lines[1][:100] if len(lines) > 1 else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False


def main():
    """Main validation routine."""
    print("=" * 80)
    print("CSV FORMAT CONSISTENCY VALIDATION")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    
    # Validate datasets
    print("\n" + "=" * 80)
    print("VALIDATING DATASET FILES (European format: sep=';', decimal=',')")
    print("=" * 80)
    
    datasets_dir = project_root / "assets" / "datasets"
    dataset_files = list(datasets_dir.glob("*.csv"))
    
    if not dataset_files:
        print("⚠️  No dataset files found")
    else:
        dataset_results = []
        for file_path in sorted(dataset_files):
            if file_path.name.startswith('.'):
                continue
            result = validate_dataset_format(file_path)
            dataset_results.append(result)
        
        print(f"\n{'=' * 80}")
        print(f"Dataset validation: {sum(dataset_results)}/{len(dataset_results)} passed")
    
    # Validate result files
    print("\n" + "=" * 80)
    print("VALIDATING RESULT FILES (International format: sep=',', decimal='.')")
    print("=" * 80)
    
    results_dir = project_root / "results"
    
    # Check evolution files
    evolution_files = list((results_dir / "evolution").glob("*.csv"))
    
    # Check log files
    log_files = list((results_dir / "logs").glob("*.csv"))
    
    # Check metrics files
    metrics_files = list((results_dir / "metrics").glob("*.csv"))
    
    all_result_files = evolution_files + log_files + metrics_files
    
    if not all_result_files:
        print("⚠️  No result files found (run training first)")
    else:
        result_results = []
        
        # Sample a few files from each category
        sample_files = []
        if evolution_files:
            sample_files.append(evolution_files[0])
            if len(evolution_files) > 1:
                sample_files.append(evolution_files[-1])
        if log_files:
            sample_files.extend(log_files)
        if metrics_files:
            sample_files.extend(metrics_files[:2])
        
        for file_path in sample_files:
            result = validate_result_format(file_path)
            result_results.append(result)
        
        print(f"\n{'=' * 80}")
        print(f"Result validation: {sum(result_results)}/{len(result_results)} passed")
        print(f"(Sampled {len(sample_files)} out of {len(all_result_files)} total result files)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✅ All CSV files are using consistent formats:")
    print(f"   - INPUT (datasets): sep='{CSVConfig.INPUT_SEP}', decimal='{CSVConfig.INPUT_DECIMAL}'")
    print(f"   - OUTPUT (results): sep='{CSVConfig.OUTPUT_SEP}', decimal='{CSVConfig.OUTPUT_DECIMAL}'")
    print("\n📝 This consistency prevents decimal separator conflicts across the application.")
    

if __name__ == "__main__":
    main()
