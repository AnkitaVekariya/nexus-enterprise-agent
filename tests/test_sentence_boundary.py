import pytest

from nexus.realtime.adapters.tts.base import first_boundary


@pytest.mark.parametrize(
    "text, expected_chunk",
    [
        ("The price is 3.5 lakh rupees", None),  # no split inside 3.5
        ("Version v0.4 at 10.30", None),
        ("Hello. World", "Hello."),
        ("Really? Yes", "Really?"),
        ("कीमत 10 रुपये है। धन्यवाद", "कीमत 10 रुपये है।"),
        ("line one\nline two", "line one\n"),
        ("Hello.", None),  # may be "Hello.5" mid-stream; the end-of-stream flush speaks it
    ],
)
def test_first_boundary(text, expected_chunk):
    idx = first_boundary(text)
    got = text[: idx + 1] if idx >= 0 else None
    assert got == expected_chunk
