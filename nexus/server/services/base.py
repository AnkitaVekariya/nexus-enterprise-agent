"""gRPC service base helpers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from concurrent import futures
from typing import Any

import grpc

from nexus.server.pool import EnginePool, TTSReplicaPool
from nexus.server.proto import media_pb2, media_pb2_grpc

logger = logging.getLogger(__name__)


class HealthServicer(media_pb2_grpc.HealthServiceServicer):
    def __init__(self, kind: str, engine: str, *, ready: bool = True, message: str = "") -> None:
        self.kind = kind
        self.engine = engine
        self.ready = ready
        self.message = message
        self.sample_rate = 16000
        self.languages: list[str] = []

    async def Check(self, request, context):  # noqa: N802
        status = "ok" if self.ready else "loading"
        return media_pb2.HealthResponse(status=status, engine=self.engine, message=self.message)

    async def Meta(self, request, context):  # noqa: N802
        return media_pb2.MetaResponse(
            kind=self.kind,
            engine=self.engine,
            sample_rate=self.sample_rate,
            languages=self.languages,
        )


def pcm16_to_float32(pcm: bytes):
    import numpy as np

    if not pcm:
        return np.array([], dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio) -> bytes:
    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


# Every server in `nexus.server up` shares ONE event loop. Engine calls are
# blocking (seconds of GPU work), so calling them inline in an async handler
# froze STT, LID and every other TTS stream until it returned. These two helpers
# run them on worker threads instead. Engines are serialized per instance unless
# they set `thread_safe = True` (e.g. maya, which batches concurrent calls itself).
_END = object()


def _call_lock(engine) -> contextlib.AbstractContextManager:
    if getattr(engine, "thread_safe", False):
        return contextlib.nullcontext()
    return engine.__dict__.setdefault("_nexus_call_lock", threading.Lock())


async def run_engine(engine, fn, *args, **kwargs):
    """Run one blocking engine call off the event loop."""

    def call():
        with _call_lock(engine):
            return fn(*args, **kwargs)

    return await asyncio.to_thread(call)


async def stream_engine(engine, gen_fn, *args, **kwargs):
    """Drive a blocking engine generator on a worker thread, yielding as it goes.

    The generator gets ``should_stop``; it flips when the consumer goes away
    (client cancelled on barge-in), so the GPU stops instead of finishing a reply
    nobody will hear.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def push(item) -> None:
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:  # loop already closed (server shutting down)
            stop.set()

    def work() -> None:
        try:
            with _call_lock(engine):
                for item in gen_fn(*args, should_stop=stop.is_set, **kwargs):
                    if stop.is_set():
                        break
                    push(item)
        except BaseException as exc:  # re-raised on the loop side
            push(exc)
        finally:
            push(_END)

    loop.run_in_executor(None, work)
    try:
        while (item := await q.get()) is not _END:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


async def serve_grpc(
    servicers: list[tuple[Any, str]],
    *,
    host: str = "127.0.0.1",
    port: int = 50051,
) -> grpc.aio.Server:
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    for add_fn, servicer in servicers:
        add_fn(servicer, server)
    listen = f"{host}:{port}"
    server.add_insecure_port(listen)
    await server.start()
    logger.info("gRPC server listening on %s", listen)
    return server
