import numpy as np
from icarus_modem_core import (
    build_frame, scramble_bits, conv_encode,
    viterbi_decode, descramble_bits, parse_frame_bits
)

payload = b"DJ2RF"
frame_bits = build_frame(payload, frame_counter=1, flags=0)

# TX-Seite
tx_bits_scr = scramble_bits(frame_bits)
tx_bits_fec = conv_encode(tx_bits_scr)

# "Kanal": ein paar Bitfehler einstreuen
rx_bits = tx_bits_fec.copy()
flip_idx = np.random.choice(len(rx_bits), size=20, replace=False)
rx_bits[flip_idx] ^= 1

# RX-Seite
dec_bits = viterbi_decode(rx_bits)
descr_bits = descramble_bits(dec_bits)

result = parse_frame_bits(descr_bits)
print("Decode-Result:", result)
