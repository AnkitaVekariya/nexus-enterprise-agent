from nexus.realtime.languages import resolve, tts_voice_for


def test_configured_voice_wins_over_language_default():
    assert tts_voice_for("en", "hf_alpha") == "hf_alpha"
    assert tts_voice_for("hi", "hf_alpha") == "hf_alpha"


def test_language_default_when_no_voice_configured():
    assert tts_voice_for("hi", None) == resolve("hi").tts_description
