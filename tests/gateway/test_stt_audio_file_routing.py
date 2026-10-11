"""Auto-STT should include AUDIO + audio documents (e.g. Telegram .m4a), not only VOICE."""

from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _event_media_is_stt_input
from gateway.run_inbound import GatewayInboundMixin


def _event(message_type, path, mime=""):
    return MessageEvent(
        text="",
        message_type=message_type,
        media_urls=[path],
        media_types=[mime] if mime is not None else [],
    )


def test_voice_note_is_stt_input():
    ev = _event(MessageType.VOICE, "/tmp/cache/voice.ogg", "audio/ogg")
    assert _event_media_is_stt_input(ev, 0) is True


def test_audio_message_is_stt_input():
    ev = _event(MessageType.AUDIO, "/tmp/cache/clip.m4a", "audio/mp4")
    assert _event_media_is_stt_input(ev, 0) is True


def test_document_m4a_is_stt_input_by_extension():
    ev = _event(MessageType.DOCUMENT, "/tmp/cache/1_2_recording.m4a", "application/octet-stream")
    assert _event_media_is_stt_input(ev, 0) is True


def test_document_pdf_is_not_stt_input():
    ev = _event(MessageType.DOCUMENT, "/tmp/cache/1_2_notes.pdf", "application/pdf")
    assert _event_media_is_stt_input(ev, 0) is False


def test_classify_routes_audio_file_to_stt_paths():
    ev = _event(MessageType.AUDIO, "/tmp/cache/clip.m4a", "audio/mp4")
    images, audio, audio_files, videos = GatewayInboundMixin._classify_inbound_media(ev, False)
    assert audio == ["/tmp/cache/clip.m4a"]
    assert audio_files == []
    assert images == []
    assert videos == []


def test_classify_routes_document_m4a_to_stt_paths():
    ev = _event(MessageType.DOCUMENT, "/tmp/cache/1_2_recording.m4a", "application/octet-stream")
    images, audio, audio_files, videos = GatewayInboundMixin._classify_inbound_media(ev, False)
    assert audio == ["/tmp/cache/1_2_recording.m4a"]
    assert audio_files == []


def test_classify_skips_stt_when_pending_prepared():
    ev = _event(MessageType.VOICE, "/tmp/cache/voice.ogg", "audio/ogg")
    images, audio, audio_files, videos = GatewayInboundMixin._classify_inbound_media(ev, True)
    assert audio == []
    assert audio_files == []
