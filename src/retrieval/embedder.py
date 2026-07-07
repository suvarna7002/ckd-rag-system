import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key securely from your .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> list[float]:
    """
    Takes a string of text and returns a vector embedding 
    using OpenAI's text-embedding-3-small model.
    """
    # Optimization: OpenAI recommends replacing newlines with spaces for embeddings
    clean_text = text.replace("\n", " ")
    
    # Call the OpenAI API
    response = client.embeddings.create(
        input=[clean_text],
        model="text-embedding-3-small"
    )
    
    # The API returns a nested JSON object. We drill down to extract just the list of floats.
    return response.data[0].embedding