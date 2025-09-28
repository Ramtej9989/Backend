"""
Helper utility functions for the Data Insight Dashboard backend
"""

import pandas as pd
import numpy as np
import io
import os
import time
import psutil
import json
import re
from typing import Dict, Any, Union, List, Tuple, Optional
import mimetypes
from datetime import datetime

def parse_file(file_content: bytes, file_extension: str) -> pd.DataFrame:
    """
    Parse a file into a pandas DataFrame based on its extension
    
    Parameters:
    -----------
    file_content : bytes
        Raw file content
    file_extension : str
        File extension (e.g., 'csv', 'xlsx')
    
    Returns:
    --------
    pd.DataFrame
        Parsed dataframe
    
    Raises:
    -------
    ValueError
        If the file type is unsupported or parsing fails
    """
    try:
        file_extension = file_extension.lower()
        
        if file_extension == 'csv':
            # Try different encodings and delimiters for CSV
            try:
                return pd.read_csv(io.BytesIO(file_content))
            except UnicodeDecodeError:
                # Try with different encoding
                return pd.read_csv(io.BytesIO(file_content), encoding='latin1')
            except pd.errors.ParserError:
                # Try with different delimiter
                return pd.read_csv(io.BytesIO(file_content), sep=';')
                
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(io.BytesIO(file_content))
            
        elif file_extension == 'json':
            return pd.read_json(io.BytesIO(file_content))
            
        elif file_extension == 'parquet':
            return pd.read_parquet(io.BytesIO(file_content))
            
        elif file_extension == 'feather':
            return pd.read_feather(io.BytesIO(file_content))
            
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
            
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")

def create_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """
    Create a standardized API response format
    
    Parameters:
    -----------
    success : bool
        Whether the operation was successful
    data : Any, optional
        Data to include in response
    error : str, optional
        Error message if not successful
        
    Returns:
    --------
    Dict[str, Any]
        Standardized response dictionary
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
        
    if error is not None:
        response["error"] = error
        
    return response

def format_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis results for API response
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Raw analysis results
        
    Returns:
    --------
    Dict[str, Any]
        Formatted results with proper serialization
    """
    # Convert any numpy types for JSON serialization
    formatted = {}
    
    for key, value in results.items():
        formatted[key] = convert_numpy_types(value)
        
    # Add metadata
    formatted["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    return formatted

def detect_file_type(filename: str) -> str:
    """
    Detect the file type from filename or content
    
    Parameters:
    -----------
    filename : str
        Name of the file
        
    Returns:
    --------
    str
        Detected file extension
    """
    # Get extension from filename
    _, extension = os.path.splitext(filename)
    
    if extension:
        # Remove the dot and return lowercase extension
        return extension[1:].lower()
    
    # If no extension, try to guess based on mime type
    mime_type, _ = mimetypes.guess_type(filename)
    
    if mime_type:
        if mime_type == 'text/csv':
            return 'csv'
        elif mime_type == 'application/vnd.ms-excel':
            return 'xls'
        elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            return 'xlsx'
        elif mime_type == 'application/json':
            return 'json'
    
    # Default to csv if can't determine
    return 'csv'

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to standard Python types for JSON serialization
    
    Parameters:
    -----------
    obj : Any
        Object that may contain numpy types
        
    Returns:
    --------
    Any
        Object with numpy types converted to standard Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate dataset for analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "Dataset is empty"
    
    # Check if DataFrame has too many columns (might cause performance issues)
    if df.shape[1] > 500:
        return False, f"Dataset has too many columns ({df.shape[1]}). Maximum allowed is 500."
    
    # Check if DataFrame has too many rows (might cause memory issues)
    if df.shape[0] > 1000000:
        return False, f"Dataset has too many rows ({df.shape[0]}). Maximum allowed is 1,000,000."
    
    # Check if DataFrame has at least one numeric column for analysis
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return False, "Dataset must have at least one numeric column for analysis"
    
    return True, None

def generate_sample_data(n_rows: int = 100, n_cols: int = 5) -> pd.DataFrame:
    """
    Generate sample data for testing
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_cols : int
        Number of columns to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate numeric columns
    data = {f"num_{i}": np.random.normal(0, 1, n_rows) for i in range(n_cols-2)}
    
    # Add a categorical column
    categories = ['A', 'B', 'C', 'D', 'E']
    data['category'] = [categories[i % len(categories)] for i in range(n_rows)]
    
    # Add a date column
    start_date = pd.Timestamp('2020-01-01')
    data['date'] = [start_date + pd.Timedelta(days=i) for i in range(n_rows)]
    
    return pd.DataFrame(data)

def measure_execution_time(func):
    """
    Decorator to measure execution time of a function
    
    Parameters:
    -----------
    func : callable
        Function to measure
        
    Returns:
    --------
    callable
        Wrapped function that measures execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

def check_memory_usage():
    """
    Check and print current memory usage
    
    Returns:
    --------
    float
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Current memory usage: {memory_usage:.2f} MB")
    return memory_usage

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names in DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns to sanitize
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sanitized column names
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Clean column names
    clean_columns = {}
    for col in df.columns:
        # Convert to string
        col_str = str(col)
        
        # Replace spaces and special characters with underscores
        clean_col = re.sub(r'[^\w\s]', '', col_str)
        clean_col = re.sub(r'\s+', '_', clean_col)
        
        # Make lowercase
        clean_col = clean_col.lower()
        
        # Ensure name is unique
        if clean_col in clean_columns.values():
            i = 1
            while f"{clean_col}_{i}" in clean_columns.values():
                i += 1
            clean_col = f"{clean_col}_{i}"
        
        clean_columns[col] = clean_col
    
    # Rename columns
    df = df.rename(columns=clean_columns)
    
    return df

def json_serialize(obj: Any) -> str:
    """
    Serialize object to JSON string with proper handling of NumPy types
    
    Parameters:
    -----------
    obj : Any
        Object to serialize
        
    Returns:
    --------
    str
        JSON string
    """
    return json.dumps(convert_numpy_types(obj))
