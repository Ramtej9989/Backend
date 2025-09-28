import unittest
import pandas as pd
import numpy as np
import io
import json
from fastapi.testclient import TestClient
from unittest import mock

# Import the FastAPI app
from app import app, fig_to_base64

# Create test client
client = TestClient(app)

class TestAppEndpoints(unittest.TestCase):
    """Test cases for the FastAPI application endpoints."""
    
    def setUp(self):
        """Set up test data and mocks before each test."""
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'numeric_col1': [1, 2, 3, 4, 5],
            'numeric_col2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'category_col': ['A', 'B', 'A', 'C', 'B'],
            'text_col': ['This is sample text', 'Another text sample', 
                         'More sample text', 'Yet another text', 'Final sample text']
        })
        
        # Create CSV content from the DataFrame
        self.csv_content = self.sample_df.to_csv(index=False).encode('utf-8')
        
    def test_analyze_endpoint_success(self):
        """Test the /analyze endpoint with valid input."""
        # Mock the file upload
        with mock.patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_df
            
            # Mock the fig_to_base64 function to avoid actual matplotlib operations
            with mock.patch('app.fig_to_base64', return_value="mock_base64_image"):
                response = client.post(
                    "/analyze",
                    files={"file": ("test.csv", self.csv_content, "text/csv")}
                )
                
                # Check response status code
                self.assertEqual(response.status_code, 200)
                
                # Check that the response contains expected keys
                data = response.json()
                self.assertIn("shape", data)
                self.assertEqual(data["shape"], [5, 4])  # 5 rows, 4 columns
                self.assertIn("columns", data)
                self.assertEqual(len(data["columns"]), 4)
                self.assertIn("missing", data)
                
                # Check for visualization results
                self.assertIn("correlation_heatmap", data)
                self.assertIn("numeric_bar", data)
                
                # Check for KPI data
                self.assertIn("numeric_col1_kpi", data)
                self.assertEqual(data["numeric_col1_kpi"]["mean"], 3.0)
                self.assertEqual(data["numeric_col1_kpi"]["min"], 1.0)
                self.assertEqual(data["numeric_col1_kpi"]["max"], 5.0)

    def test_analyze_endpoint_error_handling(self):
        """Test the /analyze endpoint error handling."""
        # Simulate an error when reading CSV
        with mock.patch('pandas.read_csv', side_effect=Exception("Test error")):
            response = client.post(
                "/analyze",
                files={"file": ("test.csv", b"invalid,csv,content", "text/csv")}
            )
            
            # Check that we get a 200 response with error information
            # (FastAPI will return the dict with error key, not a 500 status)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"], "Test error")
    
    def test_analyze_endpoint_no_file(self):
        """Test the /analyze endpoint with no file provided."""
        response = client.post("/analyze")
        # Should return 422 Unprocessable Entity
        self.assertEqual(response.status_code, 422)


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions in the application."""
    
    def test_fig_to_base64_function(self):
        """Test the fig_to_base64 function."""
        # Create a simple matplotlib figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Test the function with the figure
        result = fig_to_base64(fig)
        
        # Check that result is a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Check that result is valid base64
        try:
            import base64
            base64.b64decode(result)
            is_valid_base64 = True
        except:
            is_valid_base64 = False
        
        self.assertTrue(is_valid_base64)
        
        # Clean up
        plt.close(fig)


# Additional test cases for utility functions from helpers.py
class TestDataHandling(unittest.TestCase):
    """Test data processing and handling functions."""
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation."""
        # Create test data with known correlation
        test_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1, 2, 3, 4, 5],  # Perfect positive correlation with a
            'c': [5, 4, 3, 2, 1]   # Perfect negative correlation with a
        })
        
        # Calculate correlation
        corr_matrix = test_df.corr()
        
        # Check expected correlations
        self.assertAlmostEqual(corr_matrix.loc['a', 'b'], 1.0)
        self.assertAlmostEqual(corr_matrix.loc['a', 'c'], -1.0)
    
    def test_numeric_column_detection(self):
        """Test detection of numeric columns."""
        # Create test data with mixed types
        test_df = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [1.1, 2.2, 3.3],
            'string': ['a', 'b', 'c'],
            'mixed': [1, 2, 'c']  # This will be detected as object, not numeric
        })
        
        # Select numeric columns
        numeric_df = test_df.select_dtypes(include="number")
        
        # Check that we detected the right columns
        self.assertEqual(set(numeric_df.columns), {'numeric1', 'numeric2'})
        self.assertEqual(len(numeric_df.columns), 2)
    
    def test_missing_value_calculation(self):
        """Test calculation of missing values."""
        # Create test data with missing values
        test_df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [1, None, 3, None, 5],
            'col3': [1, 2, 3, 4, 5]  # No missing values
        })
        
        # Calculate missing values
        missing_counts = test_df.isnull().sum().to_dict()
        
        # Check missing value counts
        self.assertEqual(missing_counts['col1'], 1)
        self.assertEqual(missing_counts['col2'], 2)
        self.assertEqual(missing_counts['col3'], 0)


# Run the tests if this file is executed directly
if __name__ == '__main__':
    unittest.main()
