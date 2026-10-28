"""gRPC adapter integration tests against mock servers."""

from concurrent import futures

import grpc
import numpy as np
import pytest

from nexus.realtime.adapters.stt.grpc import GrpcSTT
from nexus.realtime.adapters.tts.grpc import GrpcTTS
from nexus.realtime.config import STTConfig, TTSConfig
from nexus.server.engines.base import EngineMeta, TTSEngine
from nexus.server.metadata import run_context_to_metadata
from nexus.server.proto import media_pb2_grpc
from nexus.server.services.base import HealthServicer
from nexus.server.services.tts_server import TtsServicer
from nexus.tools.context import RunContext
from tests.server.fixtures.mock_grpc_servers import start_mock_server, stop_servers


@pytest.mark.asyncio
async def test_grpc_stt_transcribe():
    server = await start_mock_server("stt", 50101)
    try:
        ctx = RunContext(tenant_id="t1", user_id="u1", session_id="s1")
        adapter = GrpcSTT(
            STTConfig(provider="nexus_server", base_url="127.0.0.1:50101"),
            run_context=ctx,
        )
        # minimal PCM16 frame
        pcm = b"\x00\x01" * 1600
        text = await adapter.transcribe(pcm, language="gu")
        assert "mock transcript" in text
        assert run_context_to_metadata(ctx)
    finally:
        await stop_servers([server])


@pytest.mark.asyncio
async def test_grpc_tts_synthesize():
    server = await start_mock_server("tts", 50102)
    try:
        adapter = GrpcTTS(
            TTSConfig(provider="nexus_server", base_url="127.0.0.1:50102"),
        )
        audio = await adapter.synthesize("hello world", language="hi")
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    finally:
        await stop_servers([server])


class _RecordingTTS(TTSEngine):
    meta = EngineMeta(id="recording-test", label="rec", kind="tts")
    sample_rate = 24000

    def __init__(self):
        self.texts = []

    def synthesize(self, text, language, voice=None, *, speed=1.0, params=None):
        self.texts.append(text)
        return np.zeros(240, dtype=np.float32)


class _OnePool:
    def __init__(self, eng):
        self.eng, self.sample_rate = eng, eng.sample_rate

    def acquire(self):
        return self.eng


@pytest.mark.asyncio
async def test_grpc_tts_stream_sends_text_once():
    eng = _RecordingTTS()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=2))
    media_pb2_grpc.add_TtsServiceServicer_to_server(
        TtsServicer(_OnePool(eng), HealthServicer("tts", "rec")), server
    )
    server.add_insecure_port("127.0.0.1:50110")
    await server.start()
    try:
        adapter = GrpcTTS(TTSConfig(provider="nexus_server", base_url="127.0.0.1:50110"))

        async def deltas():
            for d in ("Hello ", "world."):
                yield d

        chunks = [c async for c in adapter.stream_synthesize(deltas(), language="en")]
        assert chunks
        assert eng.texts == ["Hello world."]
    finally:
        await server.stop(0)
