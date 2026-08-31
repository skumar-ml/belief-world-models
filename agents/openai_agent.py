import logging
import os

import backoff
import openai
from openai import OpenAI

from .base import BaseAgent

logger = logging.getLogger("agent_eval")


class OpenAIAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        # Expand ${VAR} so committed configs can reference secrets via the env
        # instead of hardcoding them (e.g. api_base/api_key for the LiteLLM gateway).
        api_base = config.get("api_base", None)
        if api_base is not None:
            api_base = os.path.expandvars(api_base)
        api_key = config.get("api_key", None)
        if api_key is not None:
            api_key = os.path.expandvars(api_key)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        # Per-request timeout so a silently-hung socket surfaces as APITimeoutError
        # (which @backoff retries) instead of blocking the eval loop forever.
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=config.get("timeout", 120),
        )

    @backoff.on_exception(
        backoff.fibo,
        # APIError is the base class for all openai SDK errors (RateLimitError,
        # APIConnectionError, APITimeoutError, InternalServerError all subclass it),
        # so it covers the retryable cases. The old tuple included openai.Timeout,
        # which in openai>=2 is a non-exception deprecation stub — listing it made
        # backoff's except-tuple raise TypeError on the first real API error.
        openai.APIError,
        # Give up immediately on 4xx client errors (BadRequestError = context-length
        # overflow / malformed request): retrying can never succeed because the
        # trajectory only grows between tries, and 8 fibo waits (~33s) per doomed
        # turn stalled the eval loop. Let it raise so the loop ends the episode
        # cleanly as an agent_error instead of burning retries.
        giveup=lambda e: isinstance(e, openai.BadRequestError),
        max_tries=8,
    )
    def __call__(self, messages) -> str:
        # extra_body forwards vendor-specific kwargs to the server (e.g. vLLM's
        # chat_template_kwargs={"enable_thinking": false} to disable Qwen3 thinking).
        # Omitted from the request entirely when unset, so non-Qwen runs are unchanged.
        extra_body = self.config.get("extra_body", None)
        # Prepend the prompt with the system message
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=messages,
            max_completion_tokens=self.config.get("max_completion_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        return response.choices[0].message.content
