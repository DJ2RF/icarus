import numpy as np
import adi
import time

PLUTO_URI   = "ip:192.168.1.59"
FREQ_HZ     = 434_500_000      # LO
FS_HZ       = 50_000           # 50 kHz
TX_GAIN_DB  = 0                # kr�ftig zum Testen
TONE_HZ     = 5_000            # 5 kHz Basisband-Ton

def main():
    sdr = adi.Pluto(PLUTO_URI)

    sdr.tx_lo = FREQ_HZ
    sdr.tx_sample_rate = FS_HZ
    sdr.tx_rf_bandwidth = FS_HZ
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB

    sdr.tx_cyclic_buffer = True

    N = 16384
    n = np.arange(N)
    tone = np.exp(1j * 2 * np.pi * TONE_HZ * n / FS_HZ).astype(np.complex64)

    # leicht normalisieren
    tone /= 1.2

    sdr.tx(tone)

    print("Sende CW-Ton bei 434.500 MHz + 5 kHz (also ~434.505 MHz). STRG+C zum Beenden.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("TX stoppen...")
        sdr.tx_destroy_buffer()
        del sdr

if __name__ == "__main__":
    main()
