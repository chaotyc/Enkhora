"""Asyncio transport: persistent connections, request ids, a background loop thread.

v1 runs over TCP. The interface (``RpcClient.request`` returning a future; servers
replying with the request's ``rid``) is what a UCX/RDMA transport would implement.
NCCL is deliberately not used: losing one peer aborts the whole communicator, which
breaks the "ask someone else" failure model.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import logging
import threading
from typing import Awaitable, Callable

import torch

from . import wire

log = logging.getLogger(__name__)


class LoopThread:
    """An asyncio event loop running in a daemon thread."""

    def __init__(self, name: str = "loop"):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    def submit(self, coro: Awaitable) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, coro: Awaitable, timeout: float | None = None):
        return self.submit(coro).result(timeout)

    def call_soon(self, fn: Callable, *args):
        self.loop.call_soon_threadsafe(fn, *args)

    def stop(self):
        if self.loop.is_closed():
            return

        async def _cancel_all():
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            self.submit(_cancel_all()).result(2)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(2)


class Conn:
    """A framed connection with a send lock (many coroutines may send)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self._lock = asyncio.Lock()
        self.closed = False
        peer = writer.get_extra_info("peername")
        self.peer = f"{peer[0]}:{peer[1]}" if peer else "?"

    async def send(self, header: dict, tensors: dict[str, torch.Tensor] | None = None):
        chunks = wire.encode(header, tensors)
        async with self._lock:
            if self.closed:
                raise ConnectionError("connection closed")
            for c in chunks:
                self.writer.write(c)
            await self.writer.drain()

    async def recv(self):
        return await wire.read_frame(self.reader)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.writer.close()
            except Exception:
                pass


async def open_conn(host: str, port: int, timeout: float = 5.0) -> Conn:
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    return Conn(reader, writer)


class RemoteError(Exception):
    pass


class RpcClient:
    """Request/response over one persistent connection, matched by ``rid``.

    Messages without a ``rid`` (server pushes) go to ``on_push``.
    """

    def __init__(self, host: str, port: int, on_push: Callable[[dict, dict], None] | None = None):
        self.host, self.port = host, port
        self.on_push = on_push
        self.conn: Conn | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._reader_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self.conn is not None and not self.conn.closed

    async def connect(self, timeout: float = 5.0):
        self.conn = await open_conn(self.host, self.port, timeout)
        self._reader_task = asyncio.get_running_loop().create_task(self._read_loop())
        return self

    async def _read_loop(self):
        conn = self.conn
        try:
            while True:
                header, tensors = await conn.recv()
                rid = header.get("rid")
                fut = self._pending.pop(rid, None) if rid is not None else None
                if fut is not None:
                    if not fut.done():
                        fut.set_result((header, tensors))
                elif self.on_push is not None:
                    try:
                        self.on_push(header, tensors)
                    except Exception:
                        log.exception("push handler failed")
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            conn.close()
            err = ConnectionError(f"lost connection to {self.host}:{self.port}")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def request(self, header: dict, tensors: dict[str, torch.Tensor] | None = None):
        if not self.connected:
            raise ConnectionError(f"not connected to {self.host}:{self.port}")
        rid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.conn.send({**header, "rid": rid}, tensors)
        except Exception:
            self._pending.pop(rid, None)
            raise
        return await fut

    async def push(self, header: dict, tensors: dict[str, torch.Tensor] | None = None):
        if not self.connected:
            raise ConnectionError("not connected")
        await self.conn.send(header, tensors)

    async def close(self):
        if self.conn:
            self.conn.close()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass


Handler = Callable[[Conn, dict, dict], Awaitable[None]]


async def serve(host: str, port: int, on_message: Handler, on_close: Callable[[Conn], None] | None = None,
                concurrent_requests: bool = True) -> asyncio.base_events.Server:
    """Start a framed server. Each message is handled in its own task (pipelining)."""

    async def _client(reader, writer):
        conn = Conn(reader, writer)
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                header, tensors = await conn.recv()
                if concurrent_requests:
                    t = asyncio.get_running_loop().create_task(_safe(conn, header, tensors))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
                else:
                    await _safe(conn, header, tensors)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            conn.close()
            if on_close:
                on_close(conn)

    async def _safe(conn, header, tensors):
        try:
            await on_message(conn, header, tensors)
        except (ConnectionError, OSError):
            pass
        except Exception as e:
            log.exception("handler error for op=%s", header.get("op"))
            if "rid" in header:
                try:
                    await conn.send({"rid": header["rid"], "ok": False, "error": repr(e)})
                except Exception:
                    pass

    return await asyncio.start_server(_client, host, port)


def server_port(server) -> int:
    return server.sockets[0].getsockname()[1]
