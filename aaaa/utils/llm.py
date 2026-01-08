import asyncio
from typing import Any, Optional, Tuple

import httpx
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.settings import PipelineSettings


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "no_proxy",
)

_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_HTTP_CLIENT_TIMEOUT: Optional[Tuple[float, float]] = None


def _get_http_client(settings: PipelineSettings) -> httpx.AsyncClient:
    global _HTTP_CLIENT, _HTTP_CLIENT_TIMEOUT
    total_timeout = max(5.0, float(settings.openai_timeout))
    connect_timeout = min(15.0, total_timeout)
    timeout_key = (total_timeout, connect_timeout)
    if _HTTP_CLIENT is None or _HTTP_CLIENT_TIMEOUT != timeout_key:
        if _HTTP_CLIENT is not None:
            _schedule_client_close(_HTTP_CLIENT)
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(total_timeout, connect=connect_timeout),
            trust_env=False,
        )
        _HTTP_CLIENT_TIMEOUT = timeout_key
    return _HTTP_CLIENT


def _schedule_client_close(client: httpx.AsyncClient) -> None:
    async def _close() -> None:
        await client.aclose()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close())
    else:
        loop.create_task(_close())


async def provision_chat_model(
    content_preview: str,
    settings: PipelineSettings,
    **kwargs: Any,
) -> BaseChatModel:
    """Mirror open-notebook's provisioning pattern with lightweight heuristics."""
    _ = _estimate_tokens(content_preview)
    http_client = _get_http_client(settings)
    backup: dict[str, str] = {}
    for key in PROXY_ENV_VARS:
        if key in os.environ:
            backup[key] = os.environ.pop(key)
    try:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            http_async_client=http_client,
            openai_proxy=None,
            max_retries=3,
            **kwargs,
        )
    finally:
        os.environ.update(backup)
