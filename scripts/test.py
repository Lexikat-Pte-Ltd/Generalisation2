from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020?"},
    ],
)

print(response)