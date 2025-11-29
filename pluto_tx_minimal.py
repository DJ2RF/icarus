import numpy as np
import adi
import time

from icarus_modem_core import (
    build_frame,
    scramble_bits,
    conv_encode,
    bytes_to_bits,
    PREAMBLE_BITS,
)

########################################
# Funk- / Modem-Parameter
########################################

PLUTO_URI   = "ip:pluto.local"

FREQ_HZ     = 434_500_000        # 434.5 MHz
FS_HZ       = 50_000             # 50 kHz Sample-Rate
SYMBOL_RATE = 500                # 500 BPSK-Symbole/s (= FEC-Bits/s)
SPS         = FS_HZ // SYMBOL_RATE  # = 100

TX_GAIN_DB  = -20                # ggf. -35 / -25 / -20 probieren

ROLL_OFF    = 0.35               # RRC Roll-Off
RRC_SPAN_SYM = 8                 # Filterspanne in Symbolen (±)

# Payload
MESSAGE = b"DJ2RF"               # 5 Bytes


########################################
# Hilfsfunktionen: RRC, BPSK, etc.
########################################

def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """
    Root-Raised-Cosine-Filter:
    beta: Roll-Off
    sps:  Samples per Symbol
    span: Anzahl Symbole (auf jeder Seite span/2), total taps = span*sps+1
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    taps = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            # Sonderfall t = ±T/(4β)
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + \
                  4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti)**2)
            taps[i] = num / den
    # Normieren auf Energie 1
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def bpsk_map(bits: np.ndarray) -> np.ndarray:
    """
    BPSK-Mapping: 0 -> +1, 1 -> -1
    """
    return np.where(bits == 0, 1.0, -1.0).astype(np.float32)


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    x = np.zeros(len(symbols) * sps, dtype=np.float32)
    x[::sps] = symbols
    return x


########################################
# Hauptprogramm
########################################

def main():
    # 1) Nutzdaten-Frame im Bit-Bereich bauen
    payload = MESSAGE
    frame_bits = build_frame(payload, frame_counter=1, flags=0)
    print(f"Frame-Bits (inkl. Preamble+Sync+Header+CRC): {len(frame_bits)}")

    # 2) Scrambler
    scr_bits = scramble_bits(frame_bits)

    # 3) Convolutional FEC (Rate 1/2, K=7)
    fec_bits = conv_encode(scr_bits)
    print(f"FEC-encodierte Bits (TX-Bitstrom): {len(fec_bits)}")

    # 4) BPSK-Symbole
    symbols = bpsk_map(fec_bits)

    # 5) RRC-Filter vorbereiten
    rrc_taps = rrc_filter(ROLL_OFF, SPS, RRC_SPAN_SYM)
    print(f"RRC-Taps: {len(rrc_taps)}")

    # 6) Pulsformung: upsample -> RRC
    up = upsample(symbols, SPS)
    tx_bb = np.convolve(up, rrc_taps, mode="same").astype(np.complex64)

    # 7) Frame mehrfach wiederholen
    num_repeats = 50  # z.B. 50 Frames in einem TX-Buffer
    tx_waveform = np.tile(tx_bb, num_repeats)

    # 8) Normalisieren
    max_abs = np.max(np.abs(tx_waveform))
    if max_abs > 0:
        tx_waveform = tx_waveform / (max_abs * 1.2)

    print(f"TX-Waveform-Länge: {len(tx_waveform)} Samples")
    print(f"Dauer eines Frames: {len(tx_bb)/FS_HZ:.3f} s")

    # 9) Pluto initialisieren
    sdr = adi.Pluto(PLUTO_URI)

    sdr.tx_lo = FREQ_HZ
    sdr.tx_sample_rate = FS_HZ
    sdr.tx_rf_bandwidth = FS_HZ
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB

    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_waveform)

    print("Modem-TX läuft (cyclic). Zum Beenden STRG+C.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("TX stoppt...")
        sdr.tx_destroy_buffer()
        del sdr


if __name__ == "__main__":
    main()
