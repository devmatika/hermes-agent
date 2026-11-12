"""Matika live-voice policy: detection and idempotent application (cached agents get it every voice turn)."""
from types import SimpleNamespace

from gateway import voice_tool_policy as vp


def test_voice_turn_detected_by_marker_key_or_header():
    assert vp.resolve_voice_tool_policy(user_message="[pipecat-voice-session]\nhallo") is not None
    assert vp.resolve_voice_tool_policy(gateway_session_key="pipecat-voice:abc:de") is not None
    assert vp.resolve_voice_tool_policy(client_header="pipecat-voice") is not None
    assert vp.resolve_voice_tool_policy(user_message="normal chat", gateway_session_key="agent:main:telegram") is None


def test_policy_applied_twice_keeps_one_voice_instruction(monkeypatch):
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [{"function": {"name": "clickup_search"}}])
    agent = SimpleNamespace(tools=[], valid_tool_names=set(), enabled_toolsets=None, ephemeral_system_prompt="base",
                            max_tokens=None)
    policy = vp.default_voice_tool_policy()
    vp.apply_voice_tool_policy_to_agent(agent, policy)
    vp.apply_voice_tool_policy_to_agent(agent, policy)
    assert agent.ephemeral_system_prompt.count("live voice mode") == 1
    assert agent.max_tokens == policy.max_tokens and "clickup_search" in agent.valid_tool_names


def test_turn_runner_applies_voice_policy_before_the_run():
    src = open("gateway/run_turn_runner.py", encoding="utf-8").read()
    assert src.index("apply_voice_tool_policy_to_agent(agent, voice_policy)") < src.index(
        "result = agent.run_conversation(api_message, **kwargs)")
