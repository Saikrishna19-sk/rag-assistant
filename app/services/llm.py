import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(context, history, question):
    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer.

Context:
{context}

Conversation History:
{history}

Question:
{question}
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content