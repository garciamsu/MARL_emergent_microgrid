#!/usr/bin/env python3
"""
Test script to verify CSV format consistency across the application.
Tests both reading and writing operations with the centralized handler.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import tempfile

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.csv_handler import (
    CSVConfig,
    read_dataset_csv,
    write_result_csv,
    read_result_csv,
    save_dataset_csv
)

def test_system_info():
    """Test system information detection."""
    print("=" * 80)
    print("System Information")
    print("=" * 80)
    info = CSVConfig.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

def test_dataset_reading():
    """Test reading dataset with European format (semicolon sep, comma decimal)."""
    print("=" * 80)
    print("Test 1: Reading Dataset CSV (European Format)")
    print("=" * 80)
    
    dataset_path = "assets/datasets/microgrid_dataset_8d_hourly_comparative_quantite.csv"
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset not found: {dataset_path}")
        return False
    
    try:
        df = read_dataset_csv(dataset_path)
        print(f"✅ Successfully read dataset")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   First 3 rows:")
        print(df.head(3))
        print(f"\n   Data types:")
        print(df.dtypes)
        return True
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def test_result_writing_reading():
    """Test writing and reading result CSV with international format."""
    print("\n" + "=" * 80)
    print("Test 2: Writing and Reading Result CSV (International Format)")
    print("=" * 80)
    
    # Create sample data
    data = {
        'episode': [0, 1, 2],
        'step': [0, 1, 2],
        'power': [123456.789, 234567.891, 345678.912],
        'reward': [-0.123, 0.456, 0.789],
        'epsilon': [0.9, 0.8, 0.7]
    }
    df_original = pd.DataFrame(data)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        write_result_csv(df_original, tmp_path)
        print(f"✅ Successfully wrote result CSV")
        
        # Read back
        df_read = read_result_csv(tmp_path)
        print(f"✅ Successfully read result CSV")
        
        # Verify content
        print(f"\n   Original data:")
        print(df_original)
        print(f"\n   Read data:")
        print(df_read)
        
        # Check format of file
        with open(tmp_path, 'r') as f:
            first_lines = [f.readline() for _ in range(3)]
        
        print(f"\n   Raw file content (first 3 lines):")
        for line in first_lines:
            print(f"   {line.strip()}")
        
        # Verify separator and decimal
        if ',' in first_lines[0] and '.' in first_lines[1]:
            print(f"\n✅ Format verification: sep=',' decimal='.' ✓")
        else:
            print(f"\n❌ Format verification failed")
            return False
        
        # Clean up
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"❌ Error in write/read test: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

def test_cross_format_compatibility():
    """Test that output format is readable by standard tools."""
    print("\n" + "=" * 80)
    print("Test 3: Cross-format Compatibility")
    print("=" * 80)
    
    data = {
        'value1': [1.234, 5.678],
        'value2': [9.012, 3.456]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Write with our handler
        write_result_csv(df, tmp_path)
        
        # Try reading with vanilla pandas (standard CSV format)
        df_vanilla = pd.read_csv(tmp_path)
        
        print(f"✅ File readable by standard pd.read_csv()")
        print(f"   Data:\n{df_vanilla}")
        
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " CSV FORMAT CONSISTENCY TEST SUITE ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = []
    
    # Test 1: System info
    test_system_info()
    
    # Test 2: Dataset reading
    results.append(("Dataset Reading", test_dataset_reading()))
    
    # Test 3: Result writing/reading
    results.append(("Result Write/Read", test_result_writing_reading()))
    
    # Test 4: Cross-format compatibility
    results.append(("Cross-format Compatibility", test_cross_format_compatibility()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print()
    if all_passed:
        print("🎉 All tests PASSED! CSV format is consistent.")
    else:
        print("⚠️  Some tests FAILED. Please review the errors above.")
    print("=" * 80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
