import asyncio

import torch

from interstitial import wire
from interstitial.transport import LoopThread, RpcClient, serve, server_port


def _roundtrip(header, tensors):
    chunks = wire.encode(header, tensors)
    blob = b"".join(bytes(c) for c in chunks)

    async def go():
        r = asyncio.StreamReader()
        r.feed_data(blob)
        r.feed_eof()
        return await wire.read_frame(r)

    return asyncio.run(go())


def test_roundtrip_dtypes_and_shapes():
    ts = {
        "f32": torch.randn(3, 5),
        "bf16": torch.randn(7, 3).to(torch.bfloat16),
        "i32": torch.arange(11, dtype=torch.int32),
        "empty": torch.empty(0, 64),
        "odd": torch.arange(3, dtype=torch.uint8),     # forces padding
        "after": torch.randn(2, 2, 2),
        "strided": torch.randn(4, 6).t(),
    }
    h, out = _roundtrip({"op": "x", "n": 1}, ts)
    assert h["op"] == "x" and h["n"] == 1
    for k, t in ts.items():
        assert out[k].dtype == t.dtype and out[k].shape == t.shape
        assert torch.equal(out[k], t), k


def test_rpc_over_tcp_matches_rids_and_pushes():
    loop = LoopThread()
    pushes = []

    async def handler(conn, h, t):
        if h["op"] == "echo":
            await asyncio.sleep(0.05 if h["slow"] else 0)
            await conn.send({"rid": h["rid"], "v": h["v"]}, {"x": t["x"] * 2})
            await conn.send({"op": "note", "v": h["v"]})

    async def main():
        srv = await serve("127.0.0.1", 0, handler)
        c = await RpcClient("127.0.0.1", server_port(srv), on_push=lambda h, t: pushes.append(h["v"])).connect()
        slow = asyncio.ensure_future(c.request({"op": "echo", "v": 1, "slow": True}, {"x": torch.ones(2)}))
        fast = await c.request({"op": "echo", "v": 2, "slow": False}, {"x": torch.ones(2)})
        s = await slow
        await asyncio.sleep(0.05)
        await c.close()
        srv.close()
        return s, fast

    (h1, t1), (h2, t2) = loop.call(main(), timeout=5)
    loop.stop()
    assert h1["v"] == 1 and h2["v"] == 2            # out-of-order replies matched by rid
    assert torch.equal(t1["x"], torch.full((2,), 2.0))
    assert sorted(pushes) == [1, 2]


def test_connection_loss_fails_pending_requests():
    loop = LoopThread()

    async def handler(conn, h, t):
        conn.close()                                # die mid-request

    async def main():
        srv = await serve("127.0.0.1", 0, handler)
        c = await RpcClient("127.0.0.1", server_port(srv)).connect()
        try:
            await asyncio.wait_for(c.request({"op": "x"}), 2)
        except ConnectionError:
            return "connection error"
        finally:
            srv.close()

    assert loop.call(main(), timeout=5) == "connection error"
    loop.stop()
