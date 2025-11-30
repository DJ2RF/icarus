import numpy as np
import adi
import time

from icarus_modem_core import (
    PREAMBLE_BITS,
    build_frame,
    scramble_bits,
    conv_encode,
    viterbi_decode,
    descramble_bits,
    parse_frame_bits,
)

########################################
# Funk- / Modem-Parameter
########################################

PLUTO_URI   = "ip:192.168.1.59"

FREQ_HZ     = 434_500_000        # 434.5 MHz
FS_HZ       = 50_000             # 50 kHz Sample-Rate
SYMBOL_RATE = 500                # 500 BPSK-Symbole/s
SPS         = FS_HZ // SYMBOL_RATE  # = 100
RX_GAIN_DB  = 15                 # moderat, ggf. 10–20 testen

ROLL_OFF     = 0.35
RRC_SPAN_SYM = 8

PHASE_CANDIDATES = np.linspace(0, 2*np.pi, 16, endpoint=False)

# Frame-Layout: Payload-Länge (muss zum TX passen!)
PAYLOAD_NBYTES = 5    # "DJ2RF"
DUMMY_PAYLOAD  = b"\x00" * PAYLOAD_NBYTES

# Korrelation auf FEC-codierter Preamble
MIN_PREAMBLE_SCORE = 70  # etwas lockerer, max = len(ENC_PREAMBLE_BITS)


########################################
# RRC-Filter, BPSK-Demod
########################################

def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """
    Root-Raised-Cosine-Filter
    beta: Roll-Off
    sps:  Samples per Symbol
    span: Anzahl Symbole (Gesamtlänge = span*sps+1 Taps)
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    taps = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + \
                  4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti)**2)
            taps[i] = num / den
    # Normierung auf Energie 1
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def bpsk_hard_decision(symbols: np.ndarray) -> np.ndarray:
    """BPSK Hard-Decision: Realteil >= 0 -> Bit 0, <0 -> Bit 1"""
    return (symbols.real < 0).astype(np.uint8)


########################################
# Vorbereitung: FEC-Frame-Länge und FEC-Preamble-Muster
########################################

def prepare_encoded_lengths_and_preamble():
    """
    WICHTIGER FIX:
    - Wir bauen ein Dummy-Frame genau wie im TX,
      scramblen es und FEC-encoden es.
    - ENC_FRAME_BITS_LEN = Gesamtlaenge des FEC-Frames
    - ENC_PREAMBLE_BITS  = die ALLERERSTEN FEC-Bits,
      die zur Preamble gehören (ohne Tail zwischendrin!).
    """
    # Dummy-Frame-Bits (Preamble+Sync+Header+Payload+CRC)
    dummy_frame_bits = build_frame(DUMMY_PAYLOAD, frame_counter=0, flags=0)

    # Scrambler über komplettes Frame (wie im TX)
    dummy_scr = scramble_bits(dummy_frame_bits)

    # FEC über komplettes Frame (wie im TX)
    dummy_fec = conv_encode(dummy_scr)

    enc_frame_len = len(dummy_fec)

    # Länge der Preamble in Info-Bits
    Lp = len(PREAMBLE_BITS)          # z.B. 64 Bits
    # FEC: Rate 1/2 -> 2 Bits pro Eingangssymbol (ohne Tail am Anfang)
    enc_preamble = dummy_fec[: 2 * Lp]

    return enc_frame_len, enc_preamble


ENC_FRAME_BITS_LEN, ENC_PREAMBLE_BITS = prepare_encoded_lengths_and_preamble()


########################################
# Korrelation im FEC-Bitstrom
########################################

def correlate_encoded_preamble(rx_bits: np.ndarray, pre_bits: np.ndarray, max_search: int = 4000):
    """
    Bit-Korrelation im FEC-Bitstrom: bestes Alignment suchen.
    Gibt (index, score) zurück.
    """
    L = len(pre_bits)
    if len(rx_bits) < L + 1:
        return None

    search_len = min(max_search, len(rx_bits) - L)
    if search_len <= 0:
        return None

    best_score = -1
    best_idx = None
    for i in range(search_len):
        window = rx_bits[i:i+L]
        score = np.sum(window == pre_bits)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


########################################
# Hauptprogramm
########################################

def main():
    print("RX: erwarte FEC-Frame-Länge (Bits):", ENC_FRAME_BITS_LEN)
    print("RX: FEC-Preamble-Länge (Bits):", len(ENC_PREAMBLE_BITS))

    sdr = adi.Pluto(PLUTO_URI)

    sdr.rx_lo = FREQ_HZ
    sdr.rx_sample_rate = FS_HZ
    sdr.rx_rf_bandwidth = FS_HZ
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
    sdr.gain_control_mode_chan0 = "manual"

    sdr.rx_buffer_size = 65536

    rrc_taps = rrc_filter(ROLL_OFF, SPS, RRC_SPAN_SYM)

    print("Starte RX-Modem, warte auf Frames... (STRG+C)")
    time.sleep(0.5)

    try:
        while True:
            iq = sdr.rx()

            # RRC-Matched Filter
            yf = np.convolve(iq, rrc_taps, mode="same")

            # Debug: Pegel
            power = np.mean(np.abs(yf)**2)
            print(f"Block-Power: {10*np.log10(power+1e-12):6.1f} dB")

            best_score = -1
            best_bits = None
            best_idx = None
            best_phase = None
            best_offset = None

            # Symbol-Offsets & Phasen durchsuchen
            for offset in range(SPS):
                syms = yf[offset::SPS]
                if syms.size < ENC_FRAME_BITS_LEN + 20:
                    continue

                for phi in PHASE_CANDIDATES:
                    rot = syms * np.exp(-1j * phi)
                    bits = bpsk_hard_decision(rot)

                    res = correlate_encoded_preamble(bits, ENC_PREAMBLE_BITS)
                    if res is None:
                        continue
                    idx, score = res

                    if score > best_score:
                        best_score  = score
                        best_bits   = bits
                        best_idx    = idx
                        best_phase  = phi
                        best_offset = offset

            if best_bits is None:
                print("Keine FEC-Preamble gefunden.")
                print("-"*60)
                time.sleep(0.2)
                continue

            print(f"Beste FEC-Preamble-Korrelation: {best_score} (max {len(ENC_PREAMBLE_BITS)})")
            print(f"Beste Phase: {best_phase*180/np.pi:6.1f} Grad, Offset={best_offset}, Startindex={best_idx}")

            if best_score < MIN_PREAMBLE_SCORE:
                print("Score unter Schwelle, überspringe Block.")
                print("-"*60)
                time.sleep(0.2)
                continue

            start = best_idx
            end   = best_idx + ENC_FRAME_BITS_LEN
            if end > len(best_bits):
                print("Nicht genug FEC-Bits im Block.")
                print("-"*60)
                time.sleep(0.2)
                continue

            fec_frame_bits = best_bits[start:end]

            # Viterbi-Decode -> Info-Bits
            dec_bits = viterbi_decode(fec_frame_bits)
            print(f"Viterbi-Output-Länge: {len(dec_bits)} Bits")

            # Descrambler
            descr_bits = descramble_bits(dec_bits)

            # Frame parsen
            result = parse_frame_bits(descr_bits)
            if result is None:
                print("Frame-Parsing/CRC fehlgeschlagen.")
                print("-"*60)
                time.sleep(0.2)
                continue

            frame_counter, flags, payload = result
            print(f"*** FRAME DECODED ***")
            print(f"Frame-Counter: {frame_counter}")
            print(f"Flags: {flags}")
            print(f"Payload (HEX): {payload.hex(' ')}")
            try:
                print(f"Payload (Text): {payload.decode('utf-8')}")
            except UnicodeDecodeError:
                print("Payload (Text): <keine gültige UTF-8>")
            print("-"*60)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("RX beendet.")
    finally:
        del sdr


if __name__ == "__main__":
    main()
