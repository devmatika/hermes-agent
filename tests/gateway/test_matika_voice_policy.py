"""Matika live-voice policy: detection and idempotent application (cached agents get it every voice turn)."""
from types import SimpleNamespace

from gateway import voice_tool_policy as vp


def test_voice_turn_detected_by_marker_key_or_header():
    assert vp.resolve_voice_tool_policy(user_message="[pipecat-voice-session]\nhallo") is not None
    assert vp.resolve_voice_tool_policy(gateway_session_key="pipecat-voice:abc:de") is not None
    assert vp.resolve_voice_tool_policy(client_header="pipecat-voice") is not None
    assert vp.resolve_voice_tool_policy(user_message="normal chat", gateway_session_key="agent:main:telegram") is None


def _tool(n):
    return {"function": {"name": n}}


def test_voice_keeps_only_spoken_toolsets_including_hidden_mcp_tools(monkeypatch):
    import model_tools
    sets = {"clickup_search": "clickup", "gcal_create": "google_workspace_tools", "onyx_search": "onyx",
            "browser_open": "browser", "terminal": "terminal", "image_generate": "image_gen"}
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", sets.get)
    # clickup_search sits behind tool search: only the concrete definitions carry it
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [_tool("clickup_search"), _tool("onyx_search")])
    monkeypatch.delenv("HERMES_VOICE_TOOLSETS", raising=False)
    agent = SimpleNamespace(tools=[_tool(n) for n in ("gcal_create", "browser_open", "terminal", "image_generate")],
                            valid_tool_names=set(), enabled_toolsets=None, ephemeral_system_prompt="", max_tokens=None)
    vp.apply_voice_tool_policy_to_agent(agent, vp.default_voice_tool_policy())
    assert agent.valid_tool_names == {"clickup_search", "gcal_create", "onyx_search"}


def test_voice_toolsets_override_and_all(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_TOOLSETS", "clickup, onyx")
    assert vp.default_voice_tool_policy().enabled_toolsets == ("clickup", "onyx")
    monkeypatch.setenv("HERMES_VOICE_TOOLSETS", "all")
    assert vp.default_voice_tool_policy().enabled_toolsets is None


def test_unknown_toolsets_never_leave_voice_without_tools(monkeypatch):
    import model_tools
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", lambda n: "browser")
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [])
    agent = SimpleNamespace(tools=[_tool("browser_open")], valid_tool_names=set(), enabled_toolsets=None,
                            ephemeral_system_prompt="", max_tokens=None)
    vp.apply_voice_tool_policy_to_agent(agent, vp.VoiceToolPolicy(enabled_toolsets=("clickup",)))
    assert [t["function"]["name"] for t in agent.tools] == ["browser_open"]


def test_policy_applied_twice_keeps_one_voice_instruction(monkeypatch):
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [{"function": {"name": "clickup_search"}}])
    monkeypatch.setattr(model_tools, "get_toolset_for_tool", lambda n: "clickup")
    agent = SimpleNamespace(tools=[], valid_tool_names=set(), enabled_toolsets=None, ephemeral_system_prompt="base",
                            max_tokens=None)
    policy = vp.VoiceToolPolicy(enabled_toolsets=("clickup",))
    vp.apply_voice_tool_policy_to_agent(agent, policy)
    vp.apply_voice_tool_policy_to_agent(agent, policy)
    assert agent.ephemeral_system_prompt.count("live voice mode") == 1
    assert agent.max_tokens == policy.max_tokens and "clickup_search" in agent.valid_tool_names


def test_turn_runner_applies_voice_policy_before_the_run():
    src = open("gateway/run_turn_runner.py", encoding="utf-8").read()
    assert src.index("apply_voice_tool_policy_to_agent(agent, voice_policy)") < src.index(
        "result = agent.run_conversation(api_message, **kwargs)")
