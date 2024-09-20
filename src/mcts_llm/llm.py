from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

import os
import openai
from dotenv import load_dotenv

load_dotenv()

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENAI_BASE_URL = None


def get_openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


def get_anthropic_api_key() -> str:
    return os.environ["ANTHROPIC_API_KEY"]


def get_fireworks_api_key() -> str:
    return os.environ["FIREWORKS_API_KEY"]


def get_together_api_key() -> str:
    return '6a51a97046d5554406c58b2898d6d8843250dd1a6ecbddbf69eb689e87315a5b'#'e941c1f7b80895954d06b013dadcbabaae7e388eb157f59da571637a4e2bc7e2'#os.environ["TOGETHER_API_KEY"]


def _get_openai_client(base_url: str | None = None) -> openai.Client:
    if base_url == TOGETHER_BASE_URL:
        api_key = get_together_api_key()
    else:
        api_key = get_openai_api_key()
    return openai.Client(api_key=api_key, base_url=base_url)


def openai_chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str,
    base_url: str | None = None,
    temperature: float = 0.8,
    **kwargs,
) -> ChatCompletion:
    client = _get_openai_client(base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response
