from openai import OpenAI

from agent import handle_request
from config import load_settings


def main():
    model, base_url, api_key = load_settings()
    client = OpenAI(api_key=api_key, base_url=base_url)
    user_message = input("Введи запрос: ")
    result = handle_request(client, model, user_message)
    print("Ответ агента:")
    print(result)


if __name__ == "__main__":
    main()