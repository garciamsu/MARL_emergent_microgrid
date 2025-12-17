#!/usr/bin/env python3
"""
csv_handler.py

Centralized CSV handling utilities for consistent format across the application.
Handles locale-specific decimal separators and ensures compatibility across Windows/Linux.
"""

import pandas as pd
import platform
import locale
from typing import Optional, Union
from pathlib import Path


class CSVConfig:
    """
    Centralized CSV configuration for consistent read/write operations.
    
    Philosophy:
    - INPUT datasets: Support both comma (,) and dot (.) as decimal separator for flexibility
    - OUTPUT files: Always use dot (.) as decimal separator for international compatibility
    - Column separator: Always use comma (,) for standard CSV format
    - Encoding: UTF-8 for cross-platform compatibility
    """
    
    # Input configuration (reading datasets)
    INPUT_SEP = ";"              # Column separator for input datasets
    INPUT_DECIMAL = ","          # Decimal separator for input datasets (European format)
    INPUT_ENCODING = "utf-8"     # Encoding for input files
    
    # Output configuration (writing results)
    OUTPUT_SEP = ";"            # Column separator for output files (standard CSV)
    OUTPUT_DECIMAL = ","         # Decimal separator for output files (international)
    OUTPUT_ENCODING = "utf-8"    # Encoding for output files
    OUTPUT_FLOAT_FORMAT = "%.6f" # Float precision for output files
    
    @staticmethod
    def get_system_info() -> dict:
        """Get system information for debugging CSV issues."""
        try:
            sys_locale = locale.getdefaultlocale()
        except Exception:
            sys_locale = ("unknown", "unknown")
        
        return {
            "platform": platform.system(),
            "locale": sys_locale,
            "decimal_point": locale.localeconv().get("decimal_point", "."),
            "thousands_sep": locale.localeconv().get("thousands_sep", ","),
        }


def read_dataset_csv(
    filepath: Union[str, Path],
    sep: Optional[str] = None,
    decimal: Optional[str] = None,
    encoding: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a dataset CSV file with consistent format handling.
    
    This function is designed for reading INPUT datasets that may use
    European format (semicolon separator, comma decimal).
    
    Args:
        filepath: Path to the CSV file
        sep: Column separator (default: use CSVConfig.INPUT_SEP)
        decimal: Decimal separator (default: use CSVConfig.INPUT_DECIMAL)
        encoding: File encoding (default: use CSVConfig.INPUT_ENCODING)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with the loaded data
        
    Example:
        >>> df = read_dataset_csv('assets/datasets/microgrid_dataset.csv')
        >>> # Automatically uses sep=';', decimal=',', encoding='utf-8'
    """
    sep = sep or CSVConfig.INPUT_SEP
    decimal = decimal or CSVConfig.INPUT_DECIMAL
    encoding = encoding or CSVConfig.INPUT_ENCODING
    
    return pd.read_csv(
        filepath,
        sep=sep,
        decimal=decimal,
        encoding=encoding,
        engine="python",  # More flexible parser for different formats
        **kwargs
    )


def write_result_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    sep: Optional[str] = None,
    decimal: Optional[str] = None,
    encoding: Optional[str] = None,
    float_format: Optional[str] = None,
    index: bool = False,
    **kwargs
) -> None:
    """
    Write a DataFrame to CSV with consistent format handling.
    
    This function is designed for writing OUTPUT files (results, evolution, metrics)
    using international format (comma separator, dot decimal) for maximum compatibility.
    
    Args:
        df: DataFrame to write
        filepath: Destination path for the CSV file
        sep: Column separator (default: use CSVConfig.OUTPUT_SEP)
        decimal: Decimal separator (default: use CSVConfig.OUTPUT_DECIMAL)
        encoding: File encoding (default: use CSVConfig.OUTPUT_ENCODING)
        float_format: Float formatting (default: use CSVConfig.OUTPUT_FLOAT_FORMAT)
        index: Whether to write row indices (default: False)
        **kwargs: Additional arguments passed to df.to_csv
        
    Example:
        >>> write_result_csv(df, 'results/evolution/episode_0.csv')
        >>> # Automatically uses sep=',', decimal='.', encoding='utf-8'
    """
    sep = sep or CSVConfig.OUTPUT_SEP
    decimal = decimal or CSVConfig.OUTPUT_DECIMAL
    encoding = encoding or CSVConfig.OUTPUT_ENCODING
    float_format = float_format or CSVConfig.OUTPUT_FLOAT_FORMAT
    
    # Ensure parent directory exists (cross-platform)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(
        filepath,
        sep=sep,
        decimal=decimal,
        encoding=encoding,
        float_format=float_format,
        index=index,
        **kwargs
    )


def read_result_csv(
    filepath: Union[str, Path],
    sep: Optional[str] = None,
    decimal: Optional[str] = None,
    encoding: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a result CSV file (evolution, metrics, logs) with consistent format handling.
    
    This function is designed for reading OUTPUT files that were created by
    write_result_csv() using international format.
    
    Args:
        filepath: Path to the CSV file
        sep: Column separator (default: use CSVConfig.OUTPUT_SEP)
        decimal: Decimal separator (default: use CSVConfig.OUTPUT_DECIMAL)
        encoding: File encoding (default: use CSVConfig.OUTPUT_ENCODING)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with the loaded data
        
    Example:
        >>> df = read_result_csv('results/evolution/episode_0.csv')
        >>> # Automatically uses sep=',', decimal='.', encoding='utf-8'
    """
    sep = sep or CSVConfig.OUTPUT_SEP
    decimal = decimal or CSVConfig.OUTPUT_DECIMAL
    encoding = encoding or CSVConfig.OUTPUT_ENCODING
    
    return pd.read_csv(
        filepath,
        sep=sep,
        decimal=decimal,
        encoding=encoding,
        **kwargs
    )


# Convenience function for backward compatibility
def save_dataset_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    use_input_format: bool = True,
    **kwargs
) -> None:
    """
    Save a DataFrame as a dataset CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Destination path
        use_input_format: If True, use INPUT format (European); if False, use OUTPUT format
        **kwargs: Additional arguments
        
    Example:
        >>> # Save with European format (for manual editing)
        >>> save_dataset_csv(df, 'assets/datasets/new_dataset.csv')
        
        >>> # Save with international format (for results)
        >>> save_dataset_csv(df, 'results/data.csv', use_input_format=False)
    """
    if use_input_format:
        write_result_csv(
            df, filepath,
            sep=CSVConfig.INPUT_SEP,
            decimal=CSVConfig.INPUT_DECIMAL,
            **kwargs
        )
    else:
        write_result_csv(df, filepath, **kwargs)
