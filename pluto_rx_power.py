import numpy as np
import adi
import time

PLUTO_URI   = "ip:192.168.1.59"
FREQ_HZ     = 434_500_000      # 434.5 MHz
FS_HZ       = 50_000
RX_GAIN_DB  = 30               # ruhig mal hoch zum Testen

def main():
    sdr = adi.Pluto(PLUTO_URI)

    sdr.rx_lo = FREQ_HZ
    sdr.rx_sample_rate = FS_HZ
    sdr.rx_rf_bandwidth = FS_HZ

    # GANZ WICHTIG: AGC aus, manueller Gain
    sdr.gain_control_mode_chan0 = "manual"
    print("RX gain control mode:", sdr.gain_control_mode_chan0)
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
    print("RX hardware gain:", sdr.rx_hardwaregain_chan0)

    sdr.rx_buffer_size = 16384

    print("Starte Power-RX, STRG+C zum Beenden.")
    try:
        while True:
            iq = sdr.rx()
            p = np.mean(np.abs(iq) ** 2)
            p_db = 10 * np.log10(p + 1e-12)
            print(f"Raw-Power: {p_db:6.1f} dB")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("Abbruch.")
    finally:
        del sdr

if __name__ == "__main__":
    main()
