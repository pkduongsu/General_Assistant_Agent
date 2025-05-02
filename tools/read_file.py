import os
import tempfile
from langchain.tools import tool

@tool("read_and_save_file", return_direct=True)
def read_and_save_file(file_path: str) -> str:
    """
    Reads the content of a file, saves it to a temporary file, and returns the path to the temporary file.
    
    Args:
        file_path (str): The path of the file to read.

    Returns:
        str: The path to the temporary file containing the content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8')
    temp_file.write(content)
    temp_file.close()

    return "File read and saved successfullly to {file_path}. Read this file to process its content."
