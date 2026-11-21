"""Fragmentation of PQ-Lite messages into BLE-sized packets, and reassembly with a timeout.

Payload budget per BLE packet: LE Data Length Extension link-layer payload of 251 bytes minus
the L2CAP basic header (4 bytes) and the ATT write-command header (3 bytes) = 244 bytes
(Bluetooth Core Specification; the 251-byte limit is in configs/channel/ble_rician.yaml).
Each fragment carries a 4-byte header: message id (1 byte), fragment index (1), fragment
count (1), flags (1), leaving 240 bytes of message data.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

FRAG_HEADER = 4
LL_PAYLOAD = 251
L2CAP_ATT_OVERHEAD = 4 + 3
APP_PAYLOAD = LL_PAYLOAD - L2CAP_ATT_OVERHEAD  # 244


def fragment(msg: bytes, msg_id: int, mtu: int = APP_PAYLOAD) -> list[bytes]:
    data_per = mtu - FRAG_HEADER
    n = max(1, -(-len(msg) // data_per))
    if n > 255:
        raise ValueError("message too large for one fragment sequence")
    return [struct.pack(">BBBB", msg_id & 0xFF, i, n, 0) + msg[i * data_per:(i + 1) * data_per]
            for i in range(n)]


def n_fragments(msg_len: int, mtu: int = APP_PAYLOAD) -> int:
    return max(1, -(-msg_len // (mtu - FRAG_HEADER)))


def ll_payload_sizes(msg_len: int, mtu: int = APP_PAYLOAD) -> list[int]:
    """Link-layer payload bytes of each packet carrying a message of msg_len bytes."""
    data_per = mtu - FRAG_HEADER
    n = n_fragments(msg_len, mtu)
    sizes = [data_per] * (n - 1) + [msg_len - data_per * (n - 1)]
    return [s + FRAG_HEADER + L2CAP_ATT_OVERHEAD for s in sizes]


@dataclass
class Reassembler:
    """Collects fragments per message id; drops a message whose fragments do not all arrive
    within ``timeout`` slots of its first fragment."""
    timeout: int = 50
    _buf: dict = field(default_factory=dict)  # msg_id -> (first_slot, count, {idx: data})

    def push(self, frag: bytes, slot: int) -> bytes | None:
        self.expire(slot)
        msg_id, idx, n, _ = struct.unpack(">BBBB", frag[:FRAG_HEADER])
        first, count, parts = self._buf.setdefault(msg_id, (slot, n, {}))
        if count != n:
            raise ValueError("inconsistent fragment count")
        parts[idx] = frag[FRAG_HEADER:]
        if len(parts) == n:
            del self._buf[msg_id]
            return b"".join(parts[i] for i in range(n))
        return None

    def expire(self, slot: int) -> list[int]:
        dead = [m for m, (first, _, _) in self._buf.items() if slot - first > self.timeout]
        for m in dead:
            del self._buf[m]
        return dead
