"""Voice tool policy for Pipecat / realtime voice sessions.

Bound only for turns that look like live voice (pipecat).
Keeps the tools required for voice operations (ClickUp, Calendar, Google Workspace, Onyx, etc.)
and exposes concrete tool definitions directly so the model can invoke them in single-turn voice calls.
"""

from __future__ import annotations

import logging
import os
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

_HOOKS_INSTALLED = False

# Live voice must answer within seconds: every tool definition is input the model reads before its first word
# (all 107 tools ≈ 50k tokens per call ≈ 10 s to first token on Matika). Voice keeps only what a spoken request
# needs. Override per host with HERMES_VOICE_TOOLSETS="a,b,c" ("all" = every tool, the old behaviour).
DEFAULT_VOICE_TOOLSETS = ("clickup", "clickup_personal", "google_workspace_tools", "onyx", "memory", "web")


@dataclass
class VoiceToolPolicy:
    max_iterations: int = 4
    turn_timeout_s: float = 60.0
    enabled_toolsets: tuple[str, ...] | None = None
    effective_tools: set[str] = field(default_factory=set)
    max_tokens: int = 1024
    ephemeral_suffix: str = (
        "You are in live voice mode. Reply in 1-2 short, conversational spoken sentences suitable for speech. "
        "When asked to view, search, or perform tasks (e.g. in ClickUp, Google Calendar, Google Workspace, or company knowledge), "
        "use the appropriate tool directly. Do not claim you cannot access tools or that you are limited to knowledge base only. "
        "Keep spoken answers natural, concise, and avoid markdown formatting like tables or long bullet lists."
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
    raw = (os.environ.get("HERMES_VOICE_TOOLSETS") or "").strip()
    if raw.lower() == "all":
        return VoiceToolPolicy()
    toolsets = tuple(t.strip() for t in raw.split(",") if t.strip()) or DEFAULT_VOICE_TOOLSETS
    return VoiceToolPolicy(enabled_toolsets=toolsets)


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
    """Apply voice runtime limits (tokens, iterations, system prompt) and ensure concrete tools are available."""
    import model_tools

    # Keep only the agent's own tools whose toolset is allowed (MCP and plugin tools included). Filtering, not
    # rebuilding: a rebuild from the static registry would drop MCP tools such as ClickUp and Onyx.
    if policy.enabled_toolsets is not None:
        allowed = {str(t).strip() for t in policy.enabled_toolsets if str(t).strip()}
        name = lambda d: str((d.get("function") or {}).get("name") or "")
        candidates = {name(d): d for d in (agent.tools or []) if name(d)}
        try:  # tools hidden behind tool search are not in agent.tools: take the concrete definitions too
            for d in model_tools.get_tool_definitions(enabled_toolsets=agent.enabled_toolsets, quiet_mode=True,
                                                      skip_tool_search_assembly=True):
                candidates.setdefault(name(d), d)
        except Exception:
            logger.debug("Voice policy could not load concrete tool definitions", exc_info=True)
        kept = [d for n, d in candidates.items() if (model_tools.get_toolset_for_tool(n) or "") in allowed]
        if policy.effective_tools:
            kept = [d for d in kept if name(d) in policy.effective_tools]
        if kept:
            agent.tools = kept
            agent.valid_tool_names = {name(d) for d in kept if name(d)}
        else:  # unknown toolset names on this host: keep every tool rather than leave voice without any
            logger.warning("Voice policy toolsets %s matched no tool; keeping all %d tools", sorted(allowed),
                           len(agent.tools or []))
    else:
        # Default voice policy: augment agent.tools with concrete raw tool definitions
        # (Google Calendar, Docs, Sheets, Drive, ClickUp, etc.) so voice turns don't require
        # multiple search/describe rounds.
        try:
            raw_defs = list(
                model_tools.get_tool_definitions(
                    enabled_toolsets=agent.enabled_toolsets,
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            )
            existing_names = {
                str((d.get("function") or {}).get("name"))
                for d in agent.tools
                if (d.get("function") or {}).get("name")
            }
            for d in raw_defs:
                name = str((d.get("function") or {}).get("name") or "")
                if name and name not in existing_names:
                    agent.tools.append(d)
                    existing_names.add(name)
            agent.valid_tool_names = existing_names
        except Exception:
            logger.debug("Could not augment agent.tools with raw_defs", exc_info=True)

    if getattr(policy, "max_tokens", None):
        try:
            agent.max_tokens = int(policy.max_tokens)
        except Exception:
            logger.debug("Unable to set agent.max_tokens for voice policy", exc_info=True)

    suffix = (policy.ephemeral_suffix or "").strip()
    existing = (getattr(agent, "ephemeral_system_prompt", None) or "").strip()
    if suffix and suffix not in existing:  # a cached agent gets the policy again on every voice turn
        agent.ephemeral_system_prompt = (
            f"{existing}\n\n{suffix}".strip() if existing else suffix
        )

    logger.warning(
        "Voice tool policy applied tools_count=%d toolsets=%s max_iterations=%s max_tokens=%s",
        len(agent.valid_tool_names),
        agent.enabled_toolsets,
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
                        list(policy.enabled_toolsets) if policy.enabled_toolsets else "all",
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
