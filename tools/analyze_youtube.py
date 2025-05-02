import re
import json
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import yt_dlp

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

@tool
def answer_question_about_youtube_video(url: str, question: str) -> str:
    """
    Answers a specific question about a YouTube video using its transcript, title, and description.

    Fetches video metadata (title, description) and transcript using yt-dlp.
    If a transcript is available, it uses an LLM to answer the provided question based on the transcript content,
    using the title and description as additional context.

    Args:
        url (str): Full YouTube video URL (or any URL yt-dlp supports).
        question (str): The specific question to answer about the video's content.

    Returns:
        str: The answer to the question based on the video's transcript,
             or a message indicating the transcript was unavailable or an error occurred.
    """
    subtitle_filename = None
    video_id = None
    try:
        # 1. Get video info (title, description) and transcript using yt-dlp
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['en'], # Prioritize English
            'writeautomaticsub': True, # Also try auto-generated captions
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True,
            'outtmpl': '%(id)s', # Base name for potential subtitle file
            'noplaylist': True,
        }

        transcript_text = None # Initialize as None to clearly indicate if found
        title = "N/A"
        description = "N/A"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get metadata and ID
            # Use ignoreerrors=True to try and get metadata even if download fails later
            info_dict = ydl.extract_info(url, download=False, process=False) # process=False avoids some errors here
            video_id = info_dict.get('id')
            title = info_dict.get('title', 'Title not found')
            description = info_dict.get('description', 'Description not found')

            if not video_id:
                 # Try extracting ID from URL as a fallback if yt-dlp fails early
                 try:
                     parsed = urlparse(url)
                     if parsed.hostname in ("www.youtube.com", "youtube.com"):
                         video_id = parse_qs(parsed.query).get("v", [None])[0]
                     elif parsed.hostname == "youtu.be":
                         video_id = parsed.path.lstrip("/")
                     if not video_id:
                          return f"Error: Could not extract video ID from URL: {url}"
                 except Exception:
                      return f"Error: Could not extract video ID from URL: {url}"


            # Construct expected subtitle filename (best guess, might include lang code later)
            subtitle_filename_base = f"{video_id}" # yt-dlp adds lang/format

            # Attempt to download (this will trigger subtitle download)
            try:
                 # Re-run extract_info with download=True to trigger download actions
                 # This is often more reliable for getting subtitles written
                 ydl.extract_info(url, download=True) # Let yt-dlp handle download logic
            except yt_dlp.utils.DownloadError as de:
                 # Log subtitle-specific errors but continue if possible
                 if "subtitles" in str(de).lower():
                     print(f"Info: Subtitle download issue for {url}: {de}")
                 else:
                     # If it's not a subtitle error, it might be more critical
                     print(f"Warning: Download error for {url}: {de}")
                     # Decide if you want to return here or proceed without transcript


            # Find the actual downloaded subtitle file (json3 format, English preferred)
            found_subtitle_file = None
            transcript_status = "not_found" # Possible values: not_found, found_but_empty, found_but_error, processed

            # List potential subtitle files matching the pattern
            potential_files = [f for f in os.listdir('.') if f.startswith(video_id) and f.endswith('.json3')]

            if potential_files:
                # Prioritize English if available
                english_file = f"{video_id}.en.json3"
                if english_file in potential_files:
                    found_subtitle_file = english_file
                else:
                    # Otherwise, take the first one found (yt-dlp usually names it based on lang)
                    found_subtitle_file = potential_files[0]

                subtitle_filename = found_subtitle_file # Store the actual found filename for cleanup
                print(f"Info: Found subtitle file: {found_subtitle_file}")

                try:
                    with open(found_subtitle_file, 'r', encoding='utf-8') as f:
                        subtitle_data = json.load(f)
                    # Extract text from json3 format
                    segments = []
                    for event in subtitle_data.get('events', []):
                        if event and 'segs' in event:
                            for seg in event['segs']:
                                if seg and 'utf8' in seg:
                                    segments.append(seg['utf8'].strip())
                    processed_text = " ".join(segments)

                    if processed_text:
                        transcript_text = processed_text # Assign only if text was extracted
                        transcript_status = "processed"
                    else:
                        # File exists but no text extracted
                        print(f"Warning: Transcript file {found_subtitle_file} found but contained no processable text.")
                        transcript_status = "found_but_empty"
                        # Keep transcript_text as None

                except json.JSONDecodeError as jde:
                    print(f"Warning: Could not parse JSON in subtitle file {found_subtitle_file}: {jde}")
                    transcript_status = "found_but_error"
                    # Keep transcript_text as None
                except Exception as e:
                    print(f"Warning: Could not read/process subtitle file {found_subtitle_file}: {e}")
                    transcript_status = "found_but_error"
                    # Keep transcript_text as None
            # else: transcript_text remains None, transcript_status remains "not_found"

        # 2. Check if transcript is available before proceeding to LLM
        if transcript_text is None:
            if transcript_status == "not_found":
                 return f"Transcript not found for video {video_id}. Cannot answer question."
            elif transcript_status == "found_but_empty":
                 return f"Transcript file found ({subtitle_filename}) but contained no text. Cannot answer question."
            elif transcript_status == "found_but_error":
                 return f"Transcript file found ({subtitle_filename}) but could not be processed. Cannot answer question."
            else: # Should not happen if transcript_text is None, but as a fallback
                 return "Transcript unavailable for an unknown reason. Cannot answer question."


        # 3. Prepare prompt for LLM Q&A
        qa_prompt_template = """
        You are an assistant designed to answer questions about a YouTube video based *only* on its provided transcript, title, and description.

        Video Title: {title}
        Video Description: {description}

        Video Transcript:
        ---
        {transcript}
        ---

        Based *only* on the information provided above (primarily the transcript), answer the following question:
        Question: {question}

        If the answer cannot be found in the transcript or the provided context, state that clearly (e.g., "The transcript does not contain information about..."). Do not make assumptions or use external knowledge. Provide a concise answer.

        Answer:
        """

        prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["title", "description", "transcript", "question"]
        )

        # 4. Query LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Or another suitable model like gemini-pro
            temperature=0.0, # Keep temperature low for factual Q&A based on context
        )

        # Create a simple chain: prompt -> llm -> output parser
        chain = prompt | llm | StrOutputParser()

        # Run the chain with the extracted info
        answer = chain.invoke({
            "title": title,
            "description": description if description else "Not Available",
            "transcript": transcript_text, # Pass the extracted transcript
            "question": question
        })

        return answer

    except yt_dlp.utils.DownloadError as e:
        # More specific error for user
        error_message = f"Error during video data/subtitle download for {url}: {e}. "
        if "video unavailable" in str(e).lower():
            error_message += "The video might be private, deleted, or unavailable in your region."
        elif "subtitles" in str(e).lower():
             error_message += "Could not fetch subtitles. They might not exist for this video in English."
        else:
             error_message += "There was a problem accessing the video data."
        return error_message
    except Exception as e:
        return f"An unexpected error occurred while processing {url}: {e}"
    finally:
        # Clean up the downloaded subtitle file if it exists and was identified
        if subtitle_filename and os.path.exists(subtitle_filename):
            try:
                os.remove(subtitle_filename)
                print(f"Cleaned up subtitle file: {subtitle_filename}")
            except Exception as e:
                print(f"Warning: Could not remove subtitle file {subtitle_filename}: {e}")
        # Attempt cleanup based on video_id if filename wasn't confirmed but ID exists
        elif video_id:
             # Check common possible names based on yt-dlp patterns
             possible_cleanup_files = [f"{video_id}.en.json3", f"{video_id}.json3"]
             for fname in possible_cleanup_files:
                  if os.path.exists(fname):
                      try:
                          os.remove(fname)
                          print(f"Cleaned up potential subtitle file: {fname}")
                      except Exception as e:
                          print(f"Warning: Could not remove potential subtitle file {fname}: {e}")


if __name__ == "__main__":
    # Test case 1: Video with likely available English subtitles
    test_url_1 = "https://www.youtube.com/watch?v=JGwWNGJdvx8" # Google I/O Keynote
    test_question_1 = "What models were mentioned in the Gemini family according to the transcript?"

    # Test case 2: Video likely without subtitles or with non-English ones
    test_url_2 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Rick Astley
    test_question_2 = "Does the transcript mention the singer giving someone up?"

    # Test case 3: Invalid URL (example)
    # test_url_3 = "https://www.youtube.com/watch?v=invalididxyz"
    # test_question_3 = "What is this video about?"

    print(f"--- Test 1: Answering Question for: {test_url_1} ---")
    print(f"Question: {test_question_1}")
    # Invoke the tool using the .invoke() method with a dictionary input
    answer1 = answer_question_about_youtube_video.invoke({"url": test_url_1, "question": test_question_1})
    print(f"\nAnswer 1:\n{answer1}")
    print("--- End of Test 1 ---")

    print(f"\n--- Test 2: Answering Question for: {test_url_2} ---")
    print(f"Question: {test_question_2}")
    # Invoke the tool using the .invoke() method with a dictionary input
    answer2 = answer_question_about_youtube_video.invoke({"url": test_url_2, "question": test_question_2})
    print(f"\nAnswer 2:\n{answer2}")
    print("--- End of Test 2 ---")

    # print(f"\n--- Test 3: Answering Question for: {test_url_3} ---")
    # print(f"Question: {test_question_3}")
    # answer3 = answer_question_about_youtube_video(test_url_3, test_question_3)
    # print(f"\nAnswer 3:\n{answer3}")
    # print("--- End of Test 3 ---")