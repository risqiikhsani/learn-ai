import requests

from dotenv import load_dotenv,dotenv_values

load_dotenv()  # take environment variables from .env.

config = dotenv_values(".env.local")
TOKEN = config["HUGGINGFACE_TOKEN"]
print(TOKEN)

# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "a real cat drinks a fanta",
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
# Show the image in the default image viewer
image.save("generated_image.png")