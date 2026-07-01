"""
LLM 请求记录器，将每次 LLM 调用的请求参数写入 working_dir/completions/ 目录。

实现方式：在 chat model 的 ainvoke/invoke 方法上打 patch，
直接拦截原始 messages，比 LangChain Callback 更可靠。

用法：
    # 入口处设置一次 working_dir
    from utils.completion_logger import set_working_dir
    set_working_dir(working_dir)

    # Agent 中标记 agent 名称
    from utils.completion_logger import log_agent

    @log_agent("StoryboardArtist")
    async def draw(self, ...):
        ...
"""

import os
import json
import time
import asyncio
import logging
import contextvars
from functools import wraps
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── ContextVar ──
_agent_name_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "completion_agent_name", default="unknown"
)
_working_dir_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "completion_working_dir", default=""
)


def set_working_dir(path: str) -> None:
    """设置全局 completions 输出根目录（只需在入口处调用一次）"""
    _working_dir_ctx.set(path)
    os.makedirs(os.path.join(path, "completions"), exist_ok=True)


def log_agent(name: str):
    """装饰器：在函数执行期间设置 agent_name ContextVar，支持同步和异步函数"""

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                token = _agent_name_ctx.set(name)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _agent_name_ctx.reset(token)

            return async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                token = _agent_name_ctx.set(name)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _agent_name_ctx.reset(token)

            return sync_wrapper

    return decorator


def _write_request_log(
    agent_name: str,
    model_name: str,
    temperature: Any,
    max_tokens: Any,
    messages: List[Dict[str, str]],
) -> None:
    """将 LLM 请求参数写入 completions 目录（由 ainvoke/invoke patch 调用）"""
    working_dir = _working_dir_ctx.get()
    if not working_dir:
        return

    try:
        record = {
            "agent": agent_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        uid = str(uuid4())[:8]
        filename = f"{ts}_{agent_name}_{uid}.json"
        filepath = os.path.join(working_dir, "completions", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.debug("LLM completion logged: %s", filepath)
    except Exception:
        logger.debug("Failed to log LLM completion", exc_info=True)


def _messages_to_dicts(messages: Any) -> List[Dict[str, str]]:
    """将 LangChain message 对象/元组/字典列表转为 {role, content} 字典列表"""
    result: List[Dict[str, str]] = []
    if not isinstance(messages, (list, tuple)):
        return result

    for msg in messages:
        if hasattr(msg, "type"):
            # LangChain BaseMessage (SystemMessage, HumanMessage, AIMessage...)
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                # 多模态消息，提取文本部分
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                content = " ".join(parts)
            result.append({"role": str(msg.type), "content": str(content)})
        elif isinstance(msg, dict):
            result.append(msg)
        elif isinstance(msg, (tuple, list)) and len(msg) == 2:
            # 元组格式: ('system', 'prompt...') 或 ('human', 'prompt...')
            result.append({"role": str(msg[0]), "content": str(msg[1])})
        else:
            result.append({"role": "unknown", "content": str(msg)})

    return result


def _patch_chat_model(model: Any) -> Any:
    """
    在 chat model 实例上 monkey-patch ainvoke / invoke，
    使每次 LLM 调用时记录 messages 到 completions 目录。

    通过创建一个 wrapper 对象来代理原始 model，
    避免直接 patch Pydantic model 实例导致的各种兼容性问题。
    """
    original_ainvoke = model.ainvoke
    original_invoke = getattr(model, "invoke", None)

    async def logging_ainvoke(input: Any, *args: Any, **kwargs: Any) -> Any:
        agent_name = _agent_name_ctx.get()
        model_name = getattr(model, "model_name", "") or getattr(model, "model", "")
        temperature = getattr(model, "temperature", None)
        max_tokens = getattr(model, "max_tokens", None)
        _write_request_log(
            agent_name, model_name, temperature, max_tokens, _messages_to_dicts(input)
        )
        return await original_ainvoke(input, *args, **kwargs)

    def logging_invoke(input: Any, *args: Any, **kwargs: Any) -> Any:
        agent_name = _agent_name_ctx.get()
        model_name = getattr(model, "model_name", "") or getattr(model, "model", "")
        temperature = getattr(model, "temperature", None)
        max_tokens = getattr(model, "max_tokens", None)
        _write_request_log(
            agent_name, model_name, temperature, max_tokens, _messages_to_dicts(input)
        )
        return original_invoke(input, *args, **kwargs)

    object.__setattr__(model, "ainvoke", logging_ainvoke)
    if original_invoke:
        object.__setattr__(model, "invoke", logging_invoke)

    return model
