import json

from openai import OpenAI


def parse_decision(raw_content):
    try:
        decision = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(decision, dict):
        return None
    if set(decision) != {"next_step", "message"}:
        return None

    next_step = decision["next_step"]
    message = decision["message"]
    if not isinstance(next_step, str) or not isinstance(message, str):
        return None

    next_step = next_step.strip()
    message = message.strip()
    if not next_step or not message:
        return None

    return {"next_step": next_step, "message": message}


def request_decision(client, model, messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        raw_content = response.choices[0].message.content
    except Exception:
        return None
    return parse_decision(raw_content)