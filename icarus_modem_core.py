# icarus_modem_core.py

import numpy as np

# =========================
# Konstanten
# =========================

PREAMBLE_BITS = np.tile(np.array([1, 0], dtype=np.uint8), 32)  # 64 Bit
SYNC_WORD = 0x1ACFFC1D
HEADER_LEN_BYTES = 2          # [frame_counter, flags]
CRC_POLY = 0x1021             # CRC-16-CCITT
CRC_INIT = 0xFFFF

# Convolutional Code: Rate 1/2, K=7, Gen-Polys (CCSDS Style)
CC_K = 7
CC_GEN = (0o133, 0o171)       # G0, G1


# =========================
# Hilfsfunktionen Bits/Bytes
# =========================

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits[:len(bits) - (len(bits) % 8)]
    return np.packbits(bits).tobytes()


# =========================
# CRC-16-CCITT
# =========================

def crc16_ccitt(data: bytes, poly: int = CRC_POLY, init: int = CRC_INIT) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def append_crc16(data: bytes) -> bytes:
    crc = crc16_ccitt(data)
    return data + crc.to_bytes(2, "big")


def check_crc16(data_with_crc: bytes) -> bool:
    if len(data_with_crc) < 2:
        return False
    data = data_with_crc[:-2]
    crc_recv = int.from_bytes(data_with_crc[-2:], "big")
    return crc16_ccitt(data) == crc_recv


# =========================
# Scrambler (LFSR, 1 + x^14 + x^15)
# =========================

def scramble_bits(bits: np.ndarray, seed: int = 0x7FFF) -> np.ndarray:
    """
    Simple additive scrambler: XOR mit LFSR-Sequenz.
    Polynom: 1 + x^14 + x^15 (CCSDS-Style).
    """
    lfsr = seed & 0x7FFF
    out = np.empty_like(bits)
    for i in range(len(bits)):
        # Ausgabe-Bit
        s = (lfsr & 1)
        out[i] = bits[i] ^ s
        # Feedback
        new_bit = ((lfsr >> 14) ^ (lfsr >> 13)) & 1
        lfsr = ((lfsr << 1) & 0x7FFF) | new_bit
    return out


def descramble_bits(bits: np.ndarray, seed: int = 0x7FFF) -> np.ndarray:
    # Symmetrisch zum Scrambler
    return scramble_bits(bits, seed=seed)


# =========================
# Convolutional Encoder (K=7, R=1/2)
# =========================

def conv_encode(bits: np.ndarray, gen=CC_GEN, K: int = CC_K) -> np.ndarray:
    g0, g1 = gen
    shift = 0
    out_bits = []
    for b in bits:
        shift = ((shift << 1) | int(b)) & ((1 << K) - 1)
        o0 = bin(shift & g0).count("1") % 2
        o1 = bin(shift & g1).count("1") % 2
        out_bits.extend([o0, o1])
    # optional: Tailbits für Rückführung auf 0
    for _ in range(K - 1):
        shift = (shift << 1) & ((1 << K) - 1)
        o0 = bin(shift & g0).count("1") % 2
        o1 = bin(shift & g1).count("1") % 2
        out_bits.extend([o0, o1])
    return np.array(out_bits, dtype=np.uint8)


# =========================
# Viterbi-Decoder (hart)
# =========================

def viterbi_decode(rx_bits: np.ndarray, gen=CC_GEN, K: int = CC_K) -> np.ndarray:
    n_states = 1 << (K - 1)
    g0, g1 = gen

    # Anzahl Code-Bitpaare
    if len(rx_bits) % 2 != 0:
        rx_bits = rx_bits[:-1]
    n_sym = len(rx_bits) // 2

    # Trellis vorbereiten: für jeden State, Inputbit -> next_state, out_pair
    trellis = {}
    for state in range(n_states):
        trellis[state] = {}
        for inp in (0, 1):
            shift = ((state << 1) | inp) & ((1 << (K - 1)) - 1)
            full_reg = (state << 1) | inp
            o0 = bin(full_reg & g0).count("1") % 2
            o1 = bin(full_reg & g1).count("1") % 2
            trellis[state][inp] = (shift, (o0, o1))

    # Metriken
    INF = 1e9
    path_metric = np.full((n_sym + 1, n_states), INF)
    path_prev = np.full((n_sym + 1, n_states), -1, dtype=int)
    path_bit  = np.full((n_sym + 1, n_states), -1, dtype=int)

    path_metric[0, 0] = 0.0  # Start im State 0

    for t in range(n_sym):
        r0, r1 = int(rx_bits[2*t]), int(rx_bits[2*t+1])
        for state in range(n_states):
            if path_metric[t, state] >= INF:
                continue
            for inp in (0, 1):
                next_state, (o0, o1) = trellis[state][inp]
                # Hamming-Distanz für Hard-Bits
                dist = (r0 != o0) + (r1 != o1)
                m = path_metric[t, state] + dist
                if m < path_metric[t+1, next_state]:
                    path_metric[t+1, next_state] = m
                    path_prev[t+1, next_state] = state
                    path_bit[t+1, next_state]  = inp

    # Rückverfolgung: wir nehmen den besten Endstate (meist 0)
    end_state = np.argmin(path_metric[n_sym])
    bits_out = []
    state = end_state
    for t in range(n_sym, 0, -1):
        b = path_bit[t, state]
        bits_out.append(b)
        state = path_prev[t, state]

    bits_out = np.array(bits_out[::-1], dtype=np.uint8)
    # Tailbits entfernen (K-1)
    if len(bits_out) > (K - 1):
        bits_out = bits_out[:-(K - 1)]
    return bits_out


# =========================
# Sync-Word Bits
# =========================

def syncword_bits(sync: int = SYNC_WORD) -> np.ndarray:
    return bytes_to_bits(sync.to_bytes(4, "big"))


# =========================
# Frame Builder / Parser
# =========================

def build_frame(payload: bytes, frame_counter: int = 0, flags: int = 0) -> np.ndarray:
    """
    Baut ein komplettes Frame (ohne FEC, ohne Scrambler) als Bit-Array:
    [Preamble][Sync][Header][Payload][CRC]
    """
    assert 0 <= frame_counter <= 255
    assert 0 <= flags <= 255

    header = bytes([frame_counter & 0xFF, flags & 0xFF])
    data = header + payload
    data_crc = append_crc16(data)

    bits_sync   = syncword_bits()
    bits_header = bytes_to_bits(header)
    bits_pl_crc = bytes_to_bits(data_crc)

    frame_bits = np.concatenate([
        PREAMBLE_BITS,
        bits_sync,
        bits_header,
        bits_pl_crc,
    ])
    return frame_bits


def parse_frame_bits(frame_bits: np.ndarray):
    """
    Nimmt ein fertig demoduliertes Frame (Bits nach Viterbi+Descrambler),
    sucht Sync & extrahiert Header, Payload, CRC.
    Gibt (frame_counter, flags, payload_bytes) oder None bei Fehler.
    """
    # Wir gehen davon aus, dass PREAMBLE schon weg ist und fangen bei Sync an.
    # Hier zur Einfachheit: wir erwarten exakt das Layout.
    bits = frame_bits.copy()

    # Preamble vorne abschneiden (64 Bit)
    if len(bits) < 64 + 32 + HEADER_LEN_BYTES*8 + 16:
        return None
    bits = bits[64:]  # Preamble droppen

    # Sync prüfen
    bits_sync = bits[:32]
    if not np.array_equal(bits_sync, syncword_bits()):
        # in "echt" würde man Sync auch erst per Korrelation suchen
        return None

    idx = 32
    bits_header = bits[idx:idx+HEADER_LEN_BYTES*8]
    idx += HEADER_LEN_BYTES*8
    bits_payload_crc = bits[idx:]

    header_bytes = bits_to_bytes(bits_header)
    data_with_crc = bits_to_bytes(bits_payload_crc)

    if not check_crc16(data_with_crc):
        return None

    data = data_with_crc[:-2]
    frame_counter = data[0]
    flags = data[1]
    payload = data[2:]

    return frame_counter, flags, payload
