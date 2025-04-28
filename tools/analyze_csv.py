from langchain.tools import tool
import pandas as pd

@tool("analyze_csv")
def analyze_csv(file_path: str, question: str) -> str:
    """
    Reads a CSV file, analyzes it to answer a question, and returns the result or an error message.

    Args:
        file_path (str): The path to the CSV file.
        question (str): The question to analyze the CSV file for.

    Returns:
        str: The analysis result or an error message.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Basic analysis based on the question
        if "columns" in question.lower():
            return f"The CSV file contains the following columns: {', '.join(df.columns)}"
        elif "rows" in question.lower():
            return f"The CSV file contains {len(df)} rows."
        elif "summary" in question.lower():
            return f"Summary of the CSV file:\n{df.describe(include='all').to_string()}"
        else:
            return "Sorry, I can only answer questions about columns, rows, or provide a summary of the CSV file."
    except FileNotFoundError:
        return f"Error: The file at '{file_path}' was not found."
    except pd.errors.EmptyDataError:
        return "Error: The CSV file is empty."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"