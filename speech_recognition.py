from transformers import pipeline

transcriber = pipeline(model="openai/whisper-large-v2")
print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))