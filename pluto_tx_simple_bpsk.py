import numpy as np
import adi
import time

PLUTO_URI   = "ip:192.168.1.59"

FREQ_HZ     = 434_500_000
FS_HZ       = 50_000
SYMBOL_RATE = 500
SPS         = FS_HZ // SYMBOL_RATE

TX_GAIN_DB  = 0                # gerne so lassen, du bist nah dran
MESSAGE     = b"DJ2RF"

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def bpsk_map(bits: np.ndarray) -> np.ndarray:
    # 0 -> +1, 1 -> -1
    return np.where(bits == 0, 1.0, -1.0).astype(np.complex64)

def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    x = np.zeros(len(symbols) * sps, dtype=np.complex64)
    x[::sps] = symbols
    return x

def main():
    preamble = np.tile(np.array([1, 0], dtype=np.uint8), 32)
    payload_bits = bytes_to_bits(MESSAGE)

    frame_bits = np.concatenate([preamble, payload_bits])
    print("Frame-Bits gesamt:", len(frame_bits))

    syms = bpsk_map(frame_bits)
    tx_bb = upsample(syms, SPS)

    max_abs = np.max(np.abs(tx_bb))
    if max_abs > 0:
        tx_bb = tx_bb / (max_abs * 1.2)

    num_repeats = 100
    tx_waveform = np.tile(tx_bb, num_repeats)

    print("Samples pro Frame:", len(tx_bb))
    print("Gesamtlänge TX-Waveform:", len(tx_waveform))

    sdr = adi.Pluto(PLUTO_URI)

    sdr.tx_lo = FREQ_HZ
    sdr.tx_sample_rate = FS_HZ
    sdr.tx_rf_bandwidth = FS_HZ
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB

    sdr.tx_cyclic_buffer = True

    try:
        sdr.tx(tx_waveform)
        print("Sende einfache BPSK-Frames (DJ2RF). STRG+C zum Beenden.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("TX stoppen (KeyboardInterrupt)...")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        del sdr

if __name__ == "__main__":
    main()
