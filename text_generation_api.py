import requests
import os
from dotenv import load_dotenv,dotenv_values

load_dotenv()  # take environment variables from .env.

config = dotenv_values(".env.local")
TOKEN = config["HUGGINGFACE_TOKEN"]
print(TOKEN)


API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "what is a cat ? ",
})

print(output)