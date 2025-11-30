import numpy as np
import adi
import time

PLUTO_URI   = "ip:192.168.1.59"
FREQ_HZ     = 434_500_000
FS_HZ       = 50_000
TX_GAIN_DB  = 0   # kräftig für den Test

def main():
    sdr = adi.Pluto(PLUTO_URI)

    sdr.tx_lo = FREQ_HZ
    sdr.tx_sample_rate = FS_HZ
    sdr.tx_rf_bandwidth = FS_HZ
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB

    sdr.tx_cyclic_buffer = True

    N = 16384
    tx_waveform = np.ones(N, dtype=np.complex64)
    sdr.tx(tx_waveform)

    print("Sende CW bei 434.5 MHz. STRG+C zum Beenden.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("TX stoppen...")
        sdr.tx_destroy_buffer()
        del sdr

if __name__ == "__main__":
    main()
