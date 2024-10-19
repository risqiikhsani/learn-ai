import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv, dotenv_values

# Load environment variables
load_dotenv()
config = dotenv_values(".env.local")
TOKEN = config["HUGGINGFACE_TOKEN"]

# Initialize the Inference Client
client = InferenceClient(api_key=TOKEN)

def chat_with_model(user_message):
    # Prepare the messages for the chat model
    messages = [{"role": "user", "content": user_message}]
    
    # Collect the responses from the model
    response = ""
    for message in client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=500,
        stream=True,
    ):
        response += message.choices[0].delta.content  # Collect the response in a string
    
    return response.strip()

# Create a Gradio interface
interface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(placeholder="Ask me anything...", label="Your Question"),
    outputs=gr.Textbox(label="Model Response"),
    title="Chat with Meta-Llama-3",
    description="Ask questions and get answers from the Meta-Llama-3 model.",
    theme="compact",
)

# Launch the Gradio interface
interface.launch()
