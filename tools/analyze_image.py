import base64
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@tool
def analyze_image(img_path: str, question: str) -> str:
    """
    Extract text from an image file using a multimodal model.
    """
    all_text = ""
    try:
        # Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Analyze the image and answer the following question: " + question
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision-capable model
        # Call the vision-capable model with the prepared message list
        response = llm.invoke(message)

        # Append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # A butler should handle errors gracefully
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

if __name__ == "__main__":
    # Example usage
    img_path = r"C:\Users\pkduo\OneDrive\Máy tính\HF Agent Course Final\Final_Assignment_Template\Screenshot 2025-05-02 144021.png"
    question = "Review the chess position provided in the image. It is white's turn. Provide the correct next move for white which guarantees a win. Please provide your response in algebraic notation.?"
    # Invoke the tool using the recommended .invoke() method with a dictionary input
    result = analyze_image.invoke({"img_path": img_path, "question": question})
    print(result)