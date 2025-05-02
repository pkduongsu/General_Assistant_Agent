import os
import requests
import tempfile
from langchain.tools import tool

@tool("download_file")
def download_file(url: str) -> str:
    """
    Downloads a file from the given URL and saves it to a temporary location.
    Returns the path to the downloaded file.

    Args:
        url (str): The URL of the file to download.

    Returns:
        str: The path to the downloaded file.
    """
    try:
        # Make a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name

        # Write the content to the temporary file
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return "File downloaded and saved successfully to {temp_file_path}. Read this file to process its content."
    except Exception as e:
        return f"An error occurred while downloading the file: {e}"