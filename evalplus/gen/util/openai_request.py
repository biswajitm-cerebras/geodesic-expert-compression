import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    enable_thinking: bool = False,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    if model.startswith("o1-"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        temperature = 1.0  # o1 models do not support temperature

    if not enable_thinking:
        # message = message + " /no_think"
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=temperature,
        n=n,
        **kwargs
    )

def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.BadRequestError as e:
            # 400 is deterministic (e.g. prompt+completion exceeds context):
            # retrying never succeeds, so fail fast instead of looping forever.
            print(f"Bad request (400), not retrying: {e}")
            raise
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret


async def async_make_request(
    client: openai.AsyncOpenAI,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    enable_thinking: bool = False,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    if model.startswith("o1-"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        temperature = 1.0  # o1 models do not support temperature

    if not enable_thinking:
        # message = message + " /no_think"
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=temperature,
        n=n,
        **kwargs
    )


async def async_make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = await async_make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.BadRequestError as e:
            # 400 is deterministic (e.g. prompt+completion exceeds context):
            # retrying never succeeds, so fail fast instead of looping forever.
            print(f"Bad request (400), not retrying: {e}")
            raise
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret
