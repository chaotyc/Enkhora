"""Admin client for the control plane (the CLI's ``ctl`` subcommand and tests)."""

from __future__ import annotations

import asyncio

from ..transport import RpcClient


async def admin_async(addr: tuple[str, int], cmd: str, **kw) -> dict:
    c = await RpcClient(*addr).connect(timeout=3.0)
    try:
        h, _ = await asyncio.wait_for(c.request({"op": "admin", "cmd": cmd, **kw}), 10)
        return h
    finally:
        await c.close()


def admin(addr: tuple[str, int], cmd: str, **kw) -> dict:
    return asyncio.run(admin_async(addr, cmd, **kw))
