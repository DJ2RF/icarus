import numpy as np
import adi
import time

########################################
# Parameter
########################################

PLUTO_URI   = "ip:192.168.1.59"

FREQ_HZ     = 434_500_000
FS_HZ       = 50_000
SYMBOL_RATE = 500
SPS         = FS_HZ // SYMBOL_RATE  # = 100

RX_GAIN_DB  = 30   # Nahfeld, passt

# Preamble: 64 Bits "10 10 10 ..." = [1,0] * 32
PREAMBLE_BITS = np.tile(np.array([1, 0], dtype=np.uint8), 32)
PREAMBLE_LEN  = len(PREAMBLE_BITS)

PAYLOAD_NBYTES      = 5
PAYLOAD_BITS_LEN    = PAYLOAD_NBYTES * 8

# Mindestscore für "echte" Preamble (von 64):
MIN_PREAMBLE_SCORE  = 48   # erstmal 75%

# Kandidaten-Phasen für BPSK-Rotation (0°, 90°, 180°, 270°)
PHASE_CANDIDATES = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi])


########################################
# Hilfsfunktionen
########################################

def downsample(x: np.ndarray, sps: int, offset: int) -> np.ndarray:
    return x[offset::sps]


def bpsk_hard_decision(syms: np.ndarray) -> np.ndarray:
    """
    BPSK Hard-Decision:
      Realteil >= 0 -> Bit 0
      Realteil <  0 -> Bit 1
    """
    return (syms.real < 0).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits[:len(bits) - (len(bits) % 8)]
    return np.packbits(bits).tobytes()


########################################
# Hauptprogramm
########################################

def main():
    sdr = adi.Pluto(PLUTO_URI)

    # RX-Setup
    sdr.rx_lo = FREQ_HZ
    sdr.rx_sample_rate = FS_HZ
    sdr.rx_rf_bandwidth = FS_HZ

    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB

    sdr.rx_buffer_size = 65536

    print("Starte einfachen BPSK-RX mit Phasen-Sweep, warte auf DJ2RF... (STRG+C)")
    print("RX gain control mode:", sdr.gain_control_mode_chan0)
    print("RX hardware gain:", sdr.rx_hardwaregain_chan0)

    try:
        while True:
            iq = sdr.rx()

            # Gesamtleistung (Debug)
            power = np.mean(np.abs(iq)**2)
            print(f"Block-Power: {10*np.log10(power+1e-12):6.1f} dB")

            global_best_score_any    = -1
            global_best_score_valid  = -1
            global_best_payload_bits = None
            global_best_phase        = None
            global_best_offset       = None

            # Alle Offsets und Phasen durchsuchen
            for offset in range(SPS):
                syms_raw = downsample(iq, SPS, offset)

                min_len_syms = PREAMBLE_LEN + PAYLOAD_BITS_LEN + 10
                if len(syms_raw) < min_len_syms:
                    continue

                for phi in PHASE_CANDIDATES:
                    rot = syms_raw * np.exp(-1j * phi)
                    bits = bpsk_hard_decision(rot)

                    max_start = len(bits) - (PREAMBLE_LEN + PAYLOAD_BITS_LEN)
                    if max_start <= 0:
                        continue

                    for start_idx in range(max_start):
                        window = bits[start_idx:start_idx+PREAMBLE_LEN]

                        # normal vs invertiert
                        score_norm = np.sum(window == PREAMBLE_BITS)
                        score_inv  = np.sum((1 - window) == PREAMBLE_BITS)

                        if score_norm >= score_inv:
                            score = score_norm
                            inverted = False
                        else:
                            score = score_inv
                            inverted = True

                        # global bester Score (egal, ob mit Payload)
                        if score > global_best_score_any:
                            global_best_score_any = score

                        # nur wenn Score >= Schwelle -> Payload laden
                        if score >= MIN_PREAMBLE_SCORE:
                            pl_start = start_idx + PREAMBLE_LEN
                            pl_end   = pl_start + PAYLOAD_BITS_LEN
                            if pl_end > len(bits):
                                continue

                            payload_bits = bits[pl_start:pl_end].copy()
                            if inverted:
                                payload_bits = 1 - payload_bits

                            if score > global_best_score_valid:
                                global_best_score_valid  = score
                                global_best_payload_bits = payload_bits
                                global_best_phase        = phi
                                global_best_offset       = offset

            if global_best_payload_bits is None:
                print(f"Keine Preamble über Schwelle. Beste Korrelation (alle Versuche): {global_best_score_any}")
                print("-"*60)
                time.sleep(0.3)
                continue

            pl_bytes = bits_to_bytes(global_best_payload_bits)
            print(f"Beste Preamble-Korrelation (gültig): {global_best_score_valid} von {PREAMBLE_LEN}")
            print(f"Beste Phase: {global_best_phase*180/np.pi:5.1f} Grad, Offset={global_best_offset}")
            print("Payload (HEX):", pl_bytes.hex(" "))
            try:
                print("Payload (Text):", pl_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                print("Payload (Text): <keine gültige UTF-8>")
            print("-"*60)

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("RX beendet.")
    finally:
        del sdr


if __name__ == "__main__":
    main()
