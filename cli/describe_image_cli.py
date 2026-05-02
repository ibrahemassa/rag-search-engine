import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

MODEL = "gemma-3-27b-it"
system_prompt = '''Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary'''



def main():
    parser = argparse.ArgumentParser(description="Improve search query using an image")
    parser.add_argument( "--image", type=str, required=True, help="Image path",
    )
    
    parser.add_argument("--query", type=str, required=True, help="Query to rewrite using the image")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as image_file:
        img = image_file.read()

    client = genai.Client(api_key=api_key)
    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(model=MODEL, contents=parts)
    # response = (response.text or "").strip().strip('"')
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()

