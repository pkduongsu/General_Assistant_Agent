import base64
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@tool
def analyze_audio(audio_url: str, question: str) -> str:
    """
    Analyze audio data from a URL using a multimodal model.
    """
    # Fetch audio data
    try:
        # Fetch audio data
        response = httpx.get(audio_url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        audio_data = base64.b64encode(response.content).decode("utf-8")

        # Pass to LLM
        message = [
            HumanMessage(
                content = [
                    {
                        "type": "text",
                        "text": "Analyze the audio and answer the following question: " + question,
                    },
                    {
                        "type": "audio",
                        "source_type": "base64",
                        "data": audio_data,
                        "mime_type": "audio/mp3", # Assuming mp3, might need adjustment based on actual content type
                    },
                ],
            )
        ]

        llm_response = llm.invoke(message)
        return llm_response.content.strip()

    except httpx.MissingSchema as e:
        error_msg = f"Error analyzing audio: The provided URL '{audio_url}' is missing the 'http://' or 'https://' protocol. Please provide a complete URL."
        print(error_msg)
        return error_msg # Return the specific error to the agent
    except httpx.InvalidURL as e:
        error_msg = f"Error analyzing audio: The provided URL '{audio_url}' is invalid. Details: {str(e)}"
        print(error_msg)
        return error_msg # Return the specific error to the agent
    except httpx.RequestError as e:
        # Catch other httpx request errors (network issues, timeouts, 404s, etc.)
        error_msg = f"Error fetching audio from URL '{audio_url}': {str(e)}"
        print(error_msg)
        return error_msg # Return the specific error to the agent
    except Exception as e:
        # Catch other potential errors (base64 encoding, LLM invocation, etc.)
        error_msg = f"An unexpected error occurred during audio analysis: {str(e)}"
        print(error_msg)
        return error_msg # Return the specific error to the agent
    
if __name__ == "__main__":
    # Example usage
    audio_url = "https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3"
    question = "What is the main topic of this audio?"
    result = analyze_audio.invoke({"audio_url": audio_url, "question": question})
    print(result)