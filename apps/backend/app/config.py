import os

MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
