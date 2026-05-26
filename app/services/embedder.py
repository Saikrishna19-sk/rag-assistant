import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_cache = {}

def get_embedding(text: str):
    if text in _cache:
        return _cache[text]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    emb