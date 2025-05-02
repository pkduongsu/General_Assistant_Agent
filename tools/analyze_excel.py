from typing import Optional
from langchain.tools import tool
import pandas as pd
from pathlib import Path

@tool
def analyze_excel(file_path: str, question: str) -> str:
    """
    Analyzes an Excel file to answer questions about its content.
    Args:
        file_path (str): Path to the Excel file
        question (str): Question about the Excel data to analyze
    Returns:
        str: Analysis result or error message
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            return f"Error: File not found at {file_path}"
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Basic information about the data 
        total_rows = len(df)
        total_columns = len(df.columns)
        columns = list(df.columns)
        
        # Create a summary of the data
        summary = f"""
        Excel File Analysis:
        - Total rows: {total_rows}
        - Total columns: {total_columns}
        - Column names: {', '.join(columns)}
        - First few rows:\n{df.head().to_string()}
        """
        
        return summary
        
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"
