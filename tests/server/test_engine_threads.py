import asyncio
import time

import numpy as np
import pytest

from nexus.server.engines.base import EngineMeta, STTEngine, TTSEngine
from nexus.server.proto import media_pb2
from nexus.server.services.base import HealthServicer, stream_engine
from nexus.server.services.stt_server import SttServicer
from nexus.server.services.tts_server import TtsServicer


class _SlowTTS(TTSEngine):
    meta = EngineMeta(id="slow-test", label="slow", kind="tts")

    def synthesize(self, text, language, voice=None, *, speed=1.0, params=None):
        time.sleep(0.5)
        return np.zeros(240, dtype=np.float32)


class _FastSTT(STTEngine):
    meta = EngineMeta(id="fast-test", label="fast", kind="stt")

    def transcribe(self, audio, sample_rate, language):
        return "ok"


class _OnePool:
    sample_rate = 24000

    def __init__(self, eng):
        self.eng = eng

    def acquire(self):
        return self.eng


@pytest.mark.asyncio
async def test_slow_tts_does_not_block_stt():
    health = HealthServicer("x", "x")
    tts = TtsServicer(_OnePool(_SlowTTS()), health)
    stt = SttServicer(_FastSTT(), health)

    # Start the clock BEFORE the TTS call: on develop the 0.5s synth blocks the
    # loop inside the sleep below, so timing only the STT call would hide the bug.
    t0 = time.monotonic()
    tts_task = asyncio.create_task(tts.Synthesize(media_pb2.TtsRequest(text="hi"), None))
    await asyncio.sleep(0.05)  # let TTS start
    await stt.Transcribe(media_pb2.AudioFrame(pcm=b"\0\0" * 160), None)
    assert time.monotonic() - t0 < 0.3  # develop: ~0.55s, fixed: ~0.05s
    await tts_task


@pytest.mark.asyncio
async def test_stream_engine_stops_generator_when_consumer_leaves():
    produced = []

    class _Eng:
        pass

    def gen(*, should_stop):
        for i in range(50):
            if should_stop():
                return
            produced.append(i)
            time.sleep(0.01)
            yield i

    stream = stream_engine(_Eng(), gen)
    assert await stream.__anext__() == 0
    await stream.aclose()
    await asyncio.sleep(0.1)
    assert len(produced) < 50
