import os

import dotenv


def load_settings():
    dotenv.load_dotenv()
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    return model, base_url, api_key