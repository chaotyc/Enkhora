"""Frame codec: a JSON header plus raw tensor buffers. Nothing is pickled.

Frame layout::

    b"IST1" | u64 header_len | header JSON | buffers (each padded to 8 bytes)

The header carries a ``_t`` list of ``[name, dtype, shape, nbytes]`` describing the
buffers in order. Tensors decode as zero-copy views over one bytearray.
"""

from __future__ import annotations

import asyncio
import json
import struct

import torch

MAGIC = b"IST1"
_PREFIX = struct.Struct("!4sQ")
_ALIGN = 8
MAX_HEADER = 64 << 20

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}


def dtype_name(dt: torch.dtype) -> str:
    return DTYPE_NAMES[dt]


def _pad(n: int) -> int:
    return (-n) % _ALIGN


def encode(header: dict, tensors: dict[str, torch.Tensor] | None = None) -> list:
    """Return a list of bytes-like chunks to write in order."""
    specs, bufs = [], []
    for name, t in (tensors or {}).items():
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy() if t.numel() else b""
        nbytes = len(raw) if isinstance(raw, bytes) else raw.nbytes
        specs.append([name, dtype_name(t.dtype), list(t.shape), nbytes])
        bufs.append(memoryview(raw) if nbytes else b"")
        if _pad(nbytes):
            bufs.append(b"\0" * _pad(nbytes))
    h = dict(header)
    h["_t"] = specs
    hb = json.dumps(h, separators=(",", ":")).encode()
    return [_PREFIX.pack(MAGIC, len(hb)), hb, *bufs]


def decode_payload(header: dict, payload: bytearray) -> dict[str, torch.Tensor]:
    out, off = {}, 0
    for name, dt, shape, nbytes in header.pop("_t", []):
        dtype = DTYPES[dt]
        if nbytes == 0:
            out[name] = torch.empty(shape, dtype=dtype)
        else:
            u8 = torch.frombuffer(payload, dtype=torch.uint8, count=nbytes, offset=off)
            out[name] = u8.view(dtype).reshape(shape)
        off += nbytes + _pad(nbytes)
    return out


def payload_size(header: dict) -> int:
    return sum(n + _pad(n) for _, _, _, n in header.get("_t", []))


async def read_frame(reader: asyncio.StreamReader) -> tuple[dict, dict[str, torch.Tensor]]:
    prefix = await reader.readexactly(_PREFIX.size)
    magic, hlen = _PREFIX.unpack(prefix)
    if magic != MAGIC or hlen > MAX_HEADER:
        raise ConnectionError("bad frame")
    header = json.loads(await reader.readexactly(hlen))
    size = payload_size(header)
    payload = bytearray(await reader.readexactly(size)) if size else bytearray()
    return header, decode_payload(header, payload)


def frame_bytes(header: dict, tensors: dict[str, torch.Tensor] | None = None) -> int:
    return sum(len(c) if isinstance(c, bytes) else c.nbytes for c in encode(header, tensors))
