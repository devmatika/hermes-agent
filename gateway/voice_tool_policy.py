"""Voice tool policy for Pipecat / realtime voice sessions.

Bound only for turns that look like live voice (pipecat). Clears the default
api_server tool surface and injects a tiny allowlist (Tier B: Onyx MCP only).

This file is bind-mounted over the image path via docker-compose.override.yml
so the change is host-side and revertible.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("gateway.voice_tool_policy")

_VOICE_POLICY: ContextVar["VoiceToolPolicy | None"] = ContextVar(
    "hermes_voice_tool_policy", default=None
)

_VOICE_MARKER = "[pipecat-voice-session]"
_VOICE_SESSION_PREFIX = "pipecat-voice:"
_VOICE_CLIENT = "pipecat-voice"
_META_TOOLS = {"tool_search", "tool_describe", "tool_call"}
_WRITE_HINTS = (
    "create",
    "update",
    "delete",
    "write",
    "send",
    "post",
    "put",
    "remove",
    "grant",
    "share",
    "upload",
)

_HOOKS_INSTALLED = False


@dataclass
class VoiceToolPolicy:
    max_iterations: int = 2
    turn_timeout_s: float = 45.0
    enabled_toolsets: tuple[str, ...] = ("onyx",)
    effective_tools: set[str] = field(default_factory=set)
    max_tokens: int = 1024
    ephemeral_suffix: str = (
        "You are in live voice mode. Reply in 1-2 short spoken sentences. "
        "Prefer answering without tools; if you need Matika company knowledge, "
        "use Onyx search only. Never use terminal, browser, email, ClickUp, or "
        "Google Workspace tools in this mode."
    )


def current_voice_tool_policy() -> VoiceToolPolicy | None:
    _ensure_hooks()
    return _VOICE_POLICY.get()


def bind_voice_tool_policy(policy: VoiceToolPolicy | None) -> Token:
    _ensure_hooks()
    return _VOICE_POLICY.set(policy)


def reset_voice_tool_policy(token: Token) -> None:
    _VOICE_POLICY.reset(token)


def is_voice_turn(
    *,
    user_message: Any = None,
    gateway_session_key: Optional[str] = None,
    client_header: Optional[str] = None,
) -> bool:
    if client_header and str(client_header).strip().lower() == _VOICE_CLIENT:
        return True
    if gateway_session_key and str(gateway_session_key).startswith(_VOICE_SESSION_PREFIX):
        return True
    text = _message_as_text(user_message)
    if text and _VOICE_MARKER in text:
        return True
    return False


def default_voice_tool_policy() -> VoiceToolPolicy:
    return VoiceToolPolicy()


def resolve_voice_tool_policy(
    *,
    user_message: Any = None,
    gateway_session_key: Optional[str] = None,
    client_header: Optional[str] = None,
) -> VoiceToolPolicy | None:
    if not is_voice_turn(
        user_message=user_message,
        gateway_session_key=gateway_session_key,
        client_header=client_header,
    ):
        return None
    return default_voice_tool_policy()


def apply_voice_tool_policy_to_agent(agent: Any, policy: VoiceToolPolicy) -> None:
    """Replace the agent tool surface with Onyx retrieval tools only (fail closed)."""
    import model_tools

    toolsets = [str(t).strip() for t in (policy.enabled_toolsets or ()) if str(t).strip()]
    defs: list[dict[str, Any]] = []
    if toolsets:
        try:
            defs = list(
                model_tools.get_tool_definitions(
                    enabled_toolsets=toolsets,
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            )
        except TypeError:
            try:
                defs = list(
                    model_tools.get_tool_definitions(
                        enabled_toolsets=toolsets,
                        quiet_mode=True,
                    )
                )
            except Exception:
                logger.exception("Voice policy failed loading toolsets=%s", toolsets)
                defs = []
        except Exception:
            logger.exception("Voice policy failed loading toolsets=%s", toolsets)
            defs = []

    kept: list[dict[str, Any]] = []
    for d in defs:
        name = str((d.get("function") or {}).get("name") or "")
        low = name.lower()
        if not name or name in _META_TOOLS:
            continue
        if policy.effective_tools and name not in policy.effective_tools:
            continue
        if not policy.effective_tools:
            if "onyx" not in low and not low.startswith("mcp__onyx"):
                if toolsets != ["onyx"]:
                    continue
                if any(h in low for h in _WRITE_HINTS):
                    continue
            elif any(h in low for h in _WRITE_HINTS):
                continue
        elif any(h in low for h in _WRITE_HINTS):
            continue
        kept.append(d)

    if not kept:
        logger.warning(
            "Voice policy: no Onyx retrieval tools available for toolsets=%s "
            "(raw=%s); using empty tool surface",
            toolsets,
            sorted(str((d.get("function") or {}).get("name") or "") for d in defs),
        )
    defs = kept

    agent.tools = defs
    agent.valid_tool_names = {
        str((d.get("function") or {}).get("name"))
        for d in defs
        if (d.get("function") or {}).get("name")
    }
    agent.enabled_toolsets = toolsets

    if getattr(policy, "max_tokens", None):
        try:
            agent.max_tokens = int(policy.max_tokens)
        except Exception:
            logger.debug("Unable to set agent.max_tokens for voice policy", exc_info=True)

    suffix = (policy.ephemeral_suffix or "").strip()
    if suffix:
        existing = (getattr(agent, "ephemeral_system_prompt", None) or "").strip()
        agent.ephemeral_system_prompt = (
            f"{existing}\n\n{suffix}".strip() if existing else suffix
        )

    logger.warning(
        "Voice tool policy applied tools=%s toolsets=%s max_iterations=%s max_tokens=%s",
        sorted(agent.valid_tool_names),
        toolsets,
        policy.max_iterations,
        getattr(policy, "max_tokens", None),
    )


def _message_as_text(user_message: Any) -> str:
    if user_message is None:
        return ""
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts: list[str] = []
        for item in user_message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {None, "text"}:
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    if isinstance(user_message, dict):
        return str(user_message.get("content") or user_message.get("text") or "")
    return str(user_message)


def _ensure_hooks() -> None:
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    try:
        from gateway.platforms.api_server import APIServerAdapter
    except Exception:
        return

    if getattr(APIServerAdapter, "_matika_voice_policy_hooks_installed", False):
        _HOOKS_INSTALLED = True
        return

    original = APIServerAdapter._run_agent

    async def _run_agent_with_voice_policy(self, user_message, conversation_history, *args, **kwargs):
        token = None
        try:
            if current_voice_tool_policy() is None:
                policy = resolve_voice_tool_policy(
                    user_message=user_message,
                    gateway_session_key=kwargs.get("gateway_session_key"),
                )
                if policy is not None:
                    token = bind_voice_tool_policy(policy)
                    logger.warning(
                        "Bound voice tool policy session_key=%s max_iterations=%s toolsets=%s",
                        kwargs.get("gateway_session_key"),
                        policy.max_iterations,
                        list(policy.enabled_toolsets),
                    )
            return await original(self, user_message, conversation_history, *args, **kwargs)
        finally:
            if token is not None:
                reset_voice_tool_policy(token)

    APIServerAdapter._run_agent = _run_agent_with_voice_policy  # type: ignore[method-assign]
    APIServerAdapter._matika_voice_policy_hooks_installed = True
    _HOOKS_INSTALLED = True
    logger.warning("Installed Matika voice tool-policy hook on APIServerAdapter._run_agent")


try:
    _ensure_hooks()
except Exception:
    logger.debug("Voice policy hook install deferred", exc_info=True)
