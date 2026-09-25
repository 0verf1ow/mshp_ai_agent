from llm_client import request_decision


SYSTEM_PROMPT = """
Ты управляешь одним шагом агентного приложения.

Верни только JSON-объект с полями next_step и message.
Допустимые значения next_step:
- final_answer — данных достаточно, внешнее действие не нужно;
- need_user_data — не хватает данных, которые может сообщить пользователь;
- need_tool — нужны актуальные внешние данные или действие вне LLM.

Для final_answer запиши в message готовый ответ.
Для need_user_data запиши в message один конкретный уточняющий вопрос.
Для need_tool кратко укажи, какая возможность нужна, но не сообщай,
что действие уже выполнено.
Не добавляй Markdown и пояснения вне JSON.
""".strip()

TOOL_UNAVAILABLE_MESSAGE = (
    "Для этого запроса нужен инструмент, которого пока нет. Попробуй переформулировать запрос."
)
UNKNOWN_STEP_MESSAGE = "Модель выбрала неизвестный следующий шаг."
TECHNICAL_ERROR_MESSAGE = "Не удалось получить корректное решение модели."


def build_messages(user_message):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def route_decision(decision):
    next_step = decision["next_step"]
    if next_step in ("final_answer", "need_user_data"):
        return decision["message"]
    if next_step == "need_tool":
        return TOOL_UNAVAILABLE_MESSAGE
    return UNKNOWN_STEP_MESSAGE


def handle_request(client, model, user_message):
    messages = build_messages(user_message)
    decision = request_decision(client, model, messages)
    if decision is None:
        return TECHNICAL_ERROR_MESSAGE
    return route_decision(decision)